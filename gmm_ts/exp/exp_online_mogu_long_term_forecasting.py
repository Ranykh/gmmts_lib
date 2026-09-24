"""GMM-TS online (joint) training with the MoGU gate in place of the learned GatingNet.

Everything that defines the GMM-TS harness is inherited from
Exp_Online_Gating_Long_Term_Forecast: the MM-TSFlib numeric experts, the text expert
(frozen LLM input embeddings -> trainable MLP projection -> pooled forecast + prior_y),
the data pipeline, the learning rates, early stopping and the result files of test().
Only the gate and the loss change:

  GMM-TS : y_hat = sum_e softmax(GatingNet(x, latents))_e * mu_e
           loss  = MSE(y_hat, y) + sum_numeric MSE(mu_e, y)
  MoGU   : y_hat = sum_e w_e * mu_e,   w_e = (1/sigma_e^2) / sum_j (1/sigma_j^2)
           loss  = sum_e w_e * GaussianNLL(mu_e, sigma_e^2; y)

for any number of experts: every numeric expert in --model plus the --llm_model expert.

One uncertainty head per expert (MoGU's UncHead architecture), all trained jointly with
the experts (the LLM stays frozen):
  * text expert:    head on the pooled hidden layer of the text MLP, i.e. the expert's own
                    latent, as MoGU puts UncHead on the backbone features.
  * numeric expert: the pinned public MM-TSFlib models return only their forecast, so their
                    backbone features cannot be reached without editing MM-TSFlib. The head
                    therefore sees what the learned GMM-TS gate sees about that expert: the
                    input window and the expert's forecast (detached, so fitting the variance
                    cannot move the forecast).
All heads share one Adam at --unc_learning_rate (default 1e-2, the rate GMM-TS already uses
for its other freshly initialised head, the text projection MLP). Putting each head in its
owner's optimizer would train the numeric heads at the experts' 1e-4: on the monthly domains
(8 steps/epoch, <=10 epochs) their variances then never leave the initial value, and the gate
ranks experts by the text head alone. Measured on Economy pl=6, PatchTST+DLinear+GPT2,
10 epochs: at 1e-4 / 1e-3 the numeric sigma^2 stayed at ~0.7 / ~0.5 and the weights were
~uniform; at 1e-2 PatchTST (best expert) got w=0.80, DLinear 0.15, GPT2 0.04.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gmm_ts.exp.exp_online_gating_long_term_forecasting import (
    Exp_Online_Gating_Long_Term_Forecast, norm)
from gmm_ts.gating.mogu import (UncertaintyHead, aleatoric_epistemic,
                                inverse_variance_weights, mogu_loss)
from gmm_ts.utils.metrics import metric
from gmm_ts.utils.tools import EarlyStopping, visual

PROMPT = ("<|start_prompt|Make predictions about the future based on the following "
          "information: {}<|<end_prompt>|>")


class Exp_Online_MoGU_Long_Term_Forecast(Exp_Online_Gating_Long_Term_Forecast):

    def __init__(self, args):
        if args.use_amp:
            raise ValueError("agg_type=mogu does not support --use_amp")
        if args.output_attention:
            raise ValueError("agg_type=mogu does not support --output_attention")
        if args.pool_type not in ("avg", "max", "min"):
            raise ValueError("agg_type=mogu supports --pool_type avg|max|min, "
                             "got {}".format(args.pool_type))
        super().__init__(args)

    # ------------------------------------------------------------------ gate
    def _build_gating_module(self, args):
        """The MoGU gate has no parameters, so there is no GatingNet and no
        all_experts_config.csv lookup; instead every expert gets an uncertainty head."""
        self.gating_module = None
        self.expert_names = self.model_names + [args.llm_model]
        head_type = getattr(args, "unc_head_type", "mlp")
        num_in = args.seq_len * args.enc_in + args.pred_len * args.c_out
        self.num_unc_heads = nn.ModuleList(
            [UncertaintyHead(num_in, args.pred_len, head_type) for _ in self.model]
        ).to(self.device)
        # width of the text MLP's hidden layer: mlp_sizes = [d_llm, int(d_llm/8), text_emb]
        self.text_unc_head = UncertaintyHead(int(self.d_llm / 8), args.pred_len,
                                             head_type).to(self.device)
        # units that are optimised, early-stopped and checkpointed together
        self.num_units = [nn.ModuleDict({"expert": m, "unc_head": h})
                          for m, h in zip(self.model, self.num_unc_heads)]
        self.text_unit = nn.ModuleDict({"mlp": self.mlp, "unc_head": self.text_unc_head})

    def _select_optimizer_unc(self):
        # experts and text MLP keep the parent's optimizers and learning rates
        params = list(self.num_unc_heads.parameters()) + list(self.text_unc_head.parameters())
        return torch.optim.Adam(params, lr=getattr(self.args, "unc_learning_rate", 1e-2))

    def _set_train(self, mode):
        for u in self.num_units:
            u.train(mode)
        self.text_unit.train(mode)

    # --------------------------------------------------------------- forward
    def _forward_batch(self, data_set, batch_x, batch_y, batch_x_mark, batch_y_mark, index):
        """All experts on one batch (the parent's non-AMP path) plus their variances.

        Returns batch_x, mu and sigma2 of shape (B, E, H, 1), and the target y (B, H, 1).
        Expert order is self.expert_names: numeric experts in --model order, then the LLM.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        prior_y = torch.from_numpy(data_set.get_prior_y(index)).float().to(self.device)

        # text expert: frozen LLM input embeddings -> trainable MLP
        batch_text = data_set.get_text(index)
        if not self.Doc2Vec:
            prompt = [PROMPT.format(t) for t in batch_text]
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))
        else:
            prompt_embeddings = torch.tensor(
                [self.text_model.infer_vector(t) for t in batch_text]).to(self.device)
        if self.use_fullmodel:
            prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        else:
            prompt_emb = prompt_embeddings
        prompt_emb, latent_text_emb = self.mlp(prompt_emb)

        # numeric experts
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
        f_dim = -1 if self.args.features == 'MS' else 0
        num_mu = []
        for m in self.model:
            out = m(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(out, tuple):  # models that return (forecast, latent)
                out = out[0]
            num_mu.append(out[:, -self.args.pred_len:, f_dim:])

        # text forecast: pool over tokens, normalise, add the historical prior (as parent)
        if self.Doc2Vec:
            prompt_emb = prompt_emb.unsqueeze(-1)
        elif self.pool_type == "avg":
            prompt_emb = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2).unsqueeze(-1)
        elif self.pool_type == "max":
            prompt_emb = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2).unsqueeze(-1)
        else:  # "min" -- kept identical to the parent, which returns the max of the negation
            prompt_emb = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2).unsqueeze(-1)
        text_mu = norm(prompt_emb) + prior_y
        assert text_mu.shape == num_mu[0].shape, (text_mu.shape, num_mu[0].shape)

        # one variance per expert and horizon step
        b = batch_x.shape[0]
        x_flat = batch_x.reshape(b, -1)
        num_sigma2 = [head(torch.cat([x_flat, mu.detach().reshape(b, -1)], dim=1))
                      for head, mu in zip(self.num_unc_heads, num_mu)]
        if latent_text_emb.dim() == 3:  # (B, tokens, d) -> (B, d), as prepare_data_for_gating
            latent_text_emb = F.adaptive_avg_pool1d(latent_text_emb.transpose(1, 2), 1).squeeze(2)
        text_sigma2 = self.text_unc_head(latent_text_emb)

        mu = torch.stack(num_mu + [text_mu], dim=1)              # (B, E, H, 1)
        sigma2 = torch.stack(num_sigma2 + [text_sigma2], dim=1)  # (B, E, H, 1)
        y = batch_y[:, -self.args.pred_len:, f_dim:]
        return batch_x, mu, sigma2, y

    # ----------------------------------------------------------------- train
    def vali(self, vali_data, vali_loader, criterion=None):
        """Mean MoGU loss. MoGU early-stops on its training objective, not on MSE."""
        total_loss = []
        self._set_train(False)
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in vali_loader:
                _, mu, sigma2, y = self._forward_batch(vali_data, batch_x, batch_y,
                                                       batch_x_mark, batch_y_mark, index)
                weights = inverse_variance_weights(sigma2)
                total_loss.append(mogu_loss(mu, sigma2, weights, y).item())
        self._set_train(True)
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        num_paths = [os.path.join('./online_gating_checkpoints/model/', setting, name)
                     for name in self.model_names]
        text_path = os.path.join('./online_gating_checkpoints/mlp/', setting)
        for p in num_paths + [text_path]:
            os.makedirs(p, exist_ok=True)

        early_stopping = [EarlyStopping(patience=self.args.patience, verbose=True)
                          for _ in self.num_units]
        early_stopping_text = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        model_optim_unc = self._select_optimizer_unc()
        max_grad_norm = getattr(self.args, "max_grad_norm", 0)
        params = [p for u in self.num_units + [self.text_unit] for p in u.parameters()
                  if p.requires_grad]

        train_steps = len(train_loader)
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self._set_train(True)
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1
                for o in model_optim:
                    o.zero_grad()
                model_optim_mlp.zero_grad()
                model_optim_unc.zero_grad()

                _, mu, sigma2, y = self._forward_batch(train_data, batch_x, batch_y,
                                                       batch_x_mark, batch_y_mark, index)
                weights = inverse_variance_weights(sigma2)
                loss = mogu_loss(mu, sigma2, weights, y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                for o in model_optim:
                    o.step()
                model_optim_mlp.step()
                model_optim_unc.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}"
                  " (MoGU weighted Gaussian NLL)".format(
                      epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            for es, unit, path in zip(early_stopping, self.num_units, num_paths):
                es(vali_loss, unit, path)
            early_stopping_text(vali_loss, self.text_unit, text_path)
            if any(es.early_stop for es in early_stopping):
                print("Early stopping")
                break

        for unit, path in zip(self.num_units, num_paths):
            unit.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.text_unit.load_state_dict(torch.load(os.path.join(text_path, 'checkpoint.pth')))
        return self.model

    # ------------------------------------------------------------------ test
    def test(self, setting, data_flag='test', save_gating_dataset=False):
        """Same metrics and files as the parent, plus the MoGU per-expert outputs:
        expert_pred / expert_sigma2 / gate_weights (N, E, H), aleatoric / epistemic (N, H),
        and expert_names.txt giving the E order. These are in the model's (scaled) space."""
        test_data, test_loader = self._get_data(flag=data_flag)
        folder_path = './{}_online_gating_results/'.format(data_flag) + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        preds, trues = [], []
        extra = {"expert_pred": [], "expert_sigma2": [], "gate_weights": [],
                 "aleatoric": [], "epistemic": [], "true_scaled": []}
        self._set_train(False)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x, mu, sigma2, y = self._forward_batch(test_data, batch_x, batch_y,
                                                             batch_x_mark, batch_y_mark, index)
                weights = inverse_variance_weights(sigma2)
                y_hat = (weights * mu).sum(dim=1)  # (B, H, 1)
                aleatoric, epistemic = aleatoric_epistemic(mu, sigma2, weights)

                extra["expert_pred"].append(mu[..., 0].cpu().numpy())
                extra["expert_sigma2"].append(sigma2[..., 0].cpu().numpy())
                extra["gate_weights"].append(weights[..., 0].cpu().numpy())
                extra["aleatoric"].append(aleatoric[..., 0].cpu().numpy())
                extra["epistemic"].append(epistemic[..., 0].cpu().numpy())
                extra["true_scaled"].append(y[..., 0].cpu().numpy())

                outputs = y_hat.cpu().numpy()
                batch_y = y.cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inp.shape
                        inp = test_data.inverse_transform(inp.squeeze(0)).reshape(shape)
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd_ = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        dtw = -999
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        extra = {k: np.concatenate(v, axis=0) for k, v in extra.items()}
        for k, v in extra.items():
            np.save(folder_path + k + '.npy', v)
        with open(folder_path + 'expert_names.txt', 'w') as fh:
            fh.write("\n".join(self.expert_names) + "\n")

        # the numbers that say whether the gate did anything useful
        print("MoGU per-expert test MSE (scaled space) and mean gate weight:")
        for e, name in enumerate(self.expert_names):
            e_mse = float(np.mean((extra["expert_pred"][:, e] - extra["true_scaled"]) ** 2))
            print("    {:<14} mse={:.6f}  mean_w={:.3f}".format(
                name, e_mse, float(extra["gate_weights"][:, e].mean())))
        return mse

from gmm_ts.data_provider.data_loader import GatingDataset
from torch.utils.data import DataLoader
from gmm_ts.utils.tools import EarlyStopping, adjust_learning_rate, visual
from gmm_ts.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from gmm_ts.gating.Gating import GatingNet
warnings.filterwarnings('ignore')
import pandas as pd 

class Exp_Offline_Gating_Long_Term_Forecast(object):
    def __init__(self, args):
        super(Exp_Offline_Gating_Long_Term_Forecast, self).__init__()
        self.args=args
        all_expert_config_df = pd.read_csv(args.all_experts_config)
        experts = args.tsfn_experts + args.tsft_experts
        self.experiment_experts_config = {}
        for e in experts:
            expert_config = all_expert_config_df[all_expert_config_df['expert'] == e]
            expert_config = expert_config[expert_config['domain'] == args.domain]
            expert_config = expert_config[expert_config['pl'] == args.pred_len]
            self.experiment_experts_config[e] = {"latent_dim": int(expert_config['latent_dim']),
                                         "domain": expert_config['domain'].values[0],
                                         "freq": expert_config['freq'].values[0],
                                         "pl": int(expert_config['pl']),
                                         "expert_type": expert_config['expert_type'].values[0],
                                         "folder_path": expert_config['folder_path'].values[0]}
        self.model = GatingNet(args, self.experiment_experts_config)
        self.device = self._acquire_device()
        self.model.to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data_loader(self, flag):
        gating_dataset = GatingDataset(self.experiment_experts_config, flag)
        data_loader = DataLoader(
            gating_dataset,
            batch_size=self.args.batch_size,
            shuffle=flag == 'train',
            num_workers=self.args.num_workers,
            drop_last=flag != 'test'
        )
        return data_loader

    def _get_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _set_data_to_device(self, batch_data):
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.float().to(self.device)
            else:
                batch_data[k] = v.to(self.device)

        
    def _get_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE':   
            criterion = nn.L1Loss()
        elif self.args.loss == 'Huber':         
            criterion = nn.SmoothL1Loss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
    
        with torch.no_grad():
            for _, batch_data in enumerate(vali_loader):
                self._set_data_to_device(batch_data)
                
                y = batch_data["y_true"]
                
                # Forward pass
                if self.args.agg_type == 'hierarchical':
                    pred_y, yn_pred, yt_pred = self.model(batch_data)
                    loss = criterion(pred_y, y) + criterion(yn_pred, y) + criterion(yt_pred, y)
                else:
                    pred_y = self.model(batch_data)
                    loss = criterion(pred_y, y)

                # Move tensors to CPU
                pred = pred_y.detach().cpu()
                true = y.detach().cpu()

                # Compute loss
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader = self._get_data_loader(flag='train')
        vali_loader = self._get_data_loader(flag='val')
        test_loader = self._get_data_loader(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._get_optimizer()
        criterion = self._get_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader, 0):
                print("Epoch: {0} | Iter: {1}".format(epoch + 1, i + 1))
                self._set_data_to_device(batch_data)

                y = batch_data["y_true"]
                iter_count += 1
                model_optim.zero_grad()
                if self.args.agg_type == 'hierarchical':
                    y_pred, yn_pred, yt_pred = self.model(batch_data)
                    loss = 0.7*criterion(y_pred, y) + 0.15*criterion(yn_pred, y) + 0.15*criterion(yt_pred, y)
                else:
                    y_pred = self.model(batch_data)
                    loss = criterion(y_pred, y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, data_flag='test'):
        test_loader = self._get_data_loader(flag=data_flag)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./gating_checkpoints/' + setting, 'checkpoint.pth')))

        
        folder_path = './{}_offline_gating_results/'.format(data_flag) + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
             for i, batch_data in enumerate(test_loader):
                self._set_data_to_device(batch_data)
                y = batch_data["y_true"].float()
                
                if self.args.agg_type == 'hierarchical':
                    pred_y, _, _ = self.model(batch_data)
                else:
                    pred_y = self.model(batch_data)

                
                pred = pred_y.detach().cpu()
                true = y.detach().cpu()

                preds.append(pred)
                trues.append(true)
                
                if i % 20 == 0:
                    input = batch_data["x_n"].detach().cpu().numpy()
                    #if test_data.scale and self.args.inverse:
                    #    shape = input.shape
                    #    input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        
        # output metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return mse

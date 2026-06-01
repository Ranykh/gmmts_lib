import torch.nn as nn
#from gmm_ts.layers.Transformer_EncDec import Encoder, EncoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import LayerNorm
import torch
from gmm_ts.layers.Embed import DataEmbedding
import torch.nn.functional as F

class GatingNet(nn.Module):
    """
    Gating module that computes the gating weights for a set of input features.
    """

    def __init__(self, config: dict, expert_config: dict):
        """
        Initialize the Gating module.

        Args:
            config (dict): Configuration arguments
        """
        super(GatingNet, self).__init__()
        
        self.expert_shared_dim = config.gating_d_model
        self.agg_type = config.agg_type
        self.pred_len = config.pred_len
        self.expert_input_type = config.expert_input_type # "latent" or "prediction"

        
        # Input processing module
        self.input_embedding = DataEmbedding(config.enc_in, config.gating_d_model, 
                                             config.embed, config.freq,
                                           config.dropout)
        
        self.expert_config = expert_config
        # projection layer per expert to bring all latents/predictions to a shared dimension
        self.expert_projections = nn.ModuleList()
        for e, e_config in self.expert_config.items():
            if self.expert_input_type == "latent":
                expert_dim = e_config["latent_dim"]
            elif self.expert_input_type == "prediction":
                expert_dim = self.pred_len
            else:
                raise ValueError(f"Unknown expert input type: {self.expert_input_type}")
            
            self.expert_projections.append( 
                nn.Linear(expert_dim, config.gating_d_model, bias=False)
            )
  
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=config.gating_d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation)

        self.encoder = TransformerEncoder(encoder_layer, num_layers=config.e_layers, norm=LayerNorm(config.gating_d_model))
        
        # Gating token 
        self.gating_token = nn.Parameter(torch.zeros((1, config.gating_d_model)), requires_grad=True)

        # MLP head to regress the gating weights
        if config.agg_type == "direct":
            self.mlp_head = nn.Sequential(
                nn.Linear(config.gating_d_model, config.gating_d_model // 2),
                nn.ReLU(),
                nn.Linear(config.gating_d_model // 2, config.pred_len * len(self.expert_config))
            )
        elif config.agg_type == "latent":
            self.mlp_head = nn.Sequential(
                nn.Linear(config.gating_d_model, config.gating_d_model // 2),
                nn.ReLU(),
                nn.Linear(config.gating_d_model // 2, config.gating_d_model * len(self.expert_config))
            )
            self.pred_proj = nn.Linear(config.gating_d_model, config.pred_len)
        elif config.agg_type == "hierarchical":
            self.mlp_head1 = nn.Sequential(
                nn.Linear(config.gating_d_model, config.gating_d_model // 2),
                nn.ReLU(),
                nn.Linear(config.gating_d_model // 2, config.pred_len * len(self.expert_config))
            )
            self.mlp_head2 = nn.Sequential(
                nn.Linear(config.gating_d_model, config.gating_d_model // 2),
                nn.ReLU(),
                nn.Linear(config.gating_d_model // 2, config.pred_len)
            ) 
        else:
            raise ValueError(f"Unknown aggregation type: {config.agg_type}")
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

    # TODO implement the forward pass

    def agg_outputs(self, experts_y_pred, w):
        """
        Aggregate the predictions of the experts according to the gating weights.

        Args:
            experts_y_pred (torch.Tensor): Predictions from the experts, shape (batch_size, num_experts, pred_len).
            w (torch.Tensor): Gating weights, shape (batch_size, num_experts, pred_len).

        Returns:
            torch.Tensor: Aggregated predictions, shape (batch_size, pred_len).
        """
        # aggregate each time step in the prediction according to the gating weight
        experts_y_pred = experts_y_pred.squeeze(-1) # B x |E| x pred_len
        y_pred = experts_y_pred * w
        y_pred = y_pred.sum(dim=1) # B x pred_len
        return y_pred
    
    def forward(self, data: dict, return_w=False) -> torch.Tensor:
        """
        Forward pass through the Gating module.

        Args:
            data (dict): Input data dictionary containing the following keys:
                - 'x_n': Input tensor of shape (batch_size, seq_len, input_dim).
                - 'expert_latents': List of expert latent tensors, each of shape (batch_size, seq_len, expert_latent_dim).
                -
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # prepare the raw input time series token 
        x = data["x_n"] # raw input time series
        x_enc = self.input_embedding(x, None) 
        x_token = F.adaptive_avg_pool1d(x_enc.transpose(1, 2), 1).squeeze(2)
        
        # prepare the gating token
        gating_token = self.gating_token.expand(x.shape[0], -1, -1)

        # prepare the expert latents
        expert_latents = []
        experts_y_pred = []
        tsfn_experts_indices = [] # for hierarchical agg.
        tsft_experts_indices = [] # for hierarchical agg.

        
        # iterate over the expert latents/oredictions
        for i, e in enumerate(self.expert_config.keys()):
            # get the expert prediction and append to list
            e_pred_y = data[e + "_pred_y"]
            experts_y_pred.append(e_pred_y)
            if data.get(e + "_h_n") is not None:
                # get the expert latent
                if self.expert_input_type == "latent":
                    expert_h_n = data[e + "_h_n"]
                elif self.expert_input_type == "prediction":
                    expert_h_n = e_pred_y.squeeze(-1)[:, :self.pred_len]
                # project the expert latents to the shared dimension and append to the list
                expert_latents.append(self.expert_projections[i](expert_h_n))
                # append the expert index to the list of indices 
                tsfn_experts_indices.append(i)
            if data.get(e + "_h_t") is not None:
                # get the expert latent
                if self.expert_input_type == "latent":
                    expert_h_t = data[e + "_h_t"]
                elif self.expert_input_type == "prediction":
                    expert_h_t = e_pred_y.squeeze(-1)[:, :self.pred_len]
                # project the expert latents to the shared dimension and append to the list
                expert_latents.append(self.expert_projections[i](expert_h_t))
                # append the expert index to the list of indices 
                tsft_experts_indices.append(i)
                    
        experts_y_pred = torch.stack(experts_y_pred, dim=1) # B x |E| x pred_len Hn
        expert_latents = torch.stack(expert_latents, dim=1) # B x |E| x gating_d_model Ht
        x_token = x_token.unsqueeze(1) # B x 1 x gating_d_model

        # append gating token, input token and expert latents
        s = torch.cat([gating_token, x_token, expert_latents], dim=1) # batch_size x |E|+2 x gating_d_model
        gating_latent = self.encoder(s)[:, 0, :]

        if self.agg_type == "direct":
            w = self.mlp_head(gating_latent).reshape(gating_latent.shape[0], len(self.expert_config), self.pred_len)
            w = self.softmax(w) # B x |E| x pred_len
            # aggregate each time step in the prediction according to the gating weight
            y_pred = self.agg_outputs(experts_y_pred, w)

        elif self.agg_type == "latent":
            w = self.mlp_head(gating_latent).reshape(gating_latent.shape[0], len(self.expert_config), self.expert_shared_dim)
            w = self.softmax(w) # B x |E| x gating_d_model
            # aggregate each time step in the prediction according to the gating weight
            expert_latents = expert_latents * w # B x |E| x gating_d_model
            expert_latents = expert_latents.sum(dim=1) # B x gating_d_model
            # project the expert latents to the output dimension
            y_pred = self.pred_proj(expert_latents)

        elif self.agg_type == "hierarchical":
            w = self.mlp_head1(gating_latent).reshape(gating_latent.shape[0], len(self.expert_config), self.pred_len)
            # apply softmax to the gating weights within each modality 
            w_tsfn = self.softmax(w[:, tsfn_experts_indices, :])
            w_tsft = self.softmax(w[:, tsft_experts_indices, :])
            yn_pred = self.agg_outputs(experts_y_pred[:,tsfn_experts_indices, :], w_tsfn)
            yt_pred = self.agg_outputs(experts_y_pred[:,tsft_experts_indices, :], w_tsft)
            w_mm = F.sigmoid(self.mlp_head2(gating_latent).reshape(gating_latent.shape[0], self.pred_len))
            # aggregate the predictions from the two modalities
            y_pred = yn_pred * w_mm + yt_pred * (1 - w_mm)
            ### Commented out code to make all variants have the same api
            
            #y_pred = y_pred.unsqueeze(-1) # B x pred_len x 1
            #yn_pred = yn_pred.unsqueeze(-1)
            #yt_pred = yt_pred.unsqueeze(-1)
            #if return_w:
            #    # return the gating weights and the predictions
            #    return y_pred, yn_pred, yt_pred, w_mm
            #return y_pred#, yn_pred, yt_pred
        
        y_pred = y_pred.unsqueeze(-1) # B x pred_len x 1

        if return_w:
            # return the gating weights and the predictions
            return y_pred, w
        
        # return the predictions
        return y_pred
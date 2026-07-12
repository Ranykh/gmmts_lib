import os
import sys

mm_tsflib = os.environ.get("MM_TSFLIB_PATH")
if not mm_tsflib:
    raise RuntimeError("Set MM_TSFLIB_PATH to your MM-TSFlib clone")
sys.path.insert(0, os.path.abspath(mm_tsflib))

import argparse
import torch
import random
import numpy as np
import re

from gmm_ts.exp.exp_offline_gating_long_term_forecasting import Exp_Offline_Gating_Long_Term_Forecast
from gmm_ts.utils.print_args import print_gating_args

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Multi-Modal Time Series Forecasting with Gating Network')
 
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='mm_long_term_forecast')
    parser.add_argument('--checkpoints', type=str, default='./gating_checkpoints/', help='location of model checkpoints')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--agg_type', type=str, default='direct', help='the gating aggregation type: direct, latent, hierarchical or inv_var')
    parser.add_argument('--inv_var_norm', type=str, default='none', help="variance handling for agg_type=inv_var: 'none' (plain 1/sigma^2) or 'per_modality' (standardize log-variance within each modality to fix the text-vs-numeric scale gap)")
    parser.add_argument('--expert_input_type', type=str, default='latent', help='using the latents or the predictions of the experts as inputs')
    parser.add_argument('--save_name', type=str, default='result_mm_longterm_forecast', help='save name')

    # expert config 
    parser.add_argument('--tsfn_experts', type=str, default="Informer,Reformer", help='tsfn experts, separated by comma')
    parser.add_argument('--tsft_experts', type=str, default="BERT", help='tsft experts, separated by comma')
    parser.add_argument('--domain', type=str, default='Public_Health', help='domain name')
    parser.add_argument('--all_experts_config', type=str, default='all_experts_config.csv', help='path to csv file with all experts configurations')
    # gating architecture 
    ## configuration used by multiple components
    parser.add_argument('--gating_d_model', type=int, default=256, help='gating encoder input size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    ## input embedding configuration
    parser.add_argument('--enc_in', type=int, default=1, help='number of input features')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    ## Transformer Encoder configuration
    #parser.add_argument('--c_out', type=int, default=256, help='output size')
    #parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of ffn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    #domain= re.search(r'/([^/]+)$', args.root_path).group(1)
    #print("now running on domain {} model {} ".format(domain,args.model))
    args.use_gpu = True
    # Note: overriding number of features to 1, as current framework only supports uni-variate time series
    args.enc_in = 1
    fix_seed = args.seed
    print("Now using seed {}".format(fix_seed))
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_gating_args(args)

    if args.task_name == 'mm_long_term_forecast':
        Exp = Exp_Offline_Gating_Long_Term_Forecast
    else:
        raise NotImplementedError
   
    args.tsfn_experts = args.tsfn_experts.split(',') 
    args.tsft_experts = args.tsft_experts.split(',')
    tsfn_experts_str = '-'.join([f'{k}' for k in args.tsfn_experts])
    tsft_experts_str = '-'.join([f'{k}' for k in args.tsft_experts])
    domain = args.domain.replace('_', '-')
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            
            exp = Exp(args)  # set experiments
            
            setting = '{}_{}_agg{}_eip{}_pl{}_dm{}_nh{}_el{}_df{}_{}'.format(
                args.domain,
                args.task_name,
                args.agg_type,
                args.expert_input_type,
                args.pred_len,
                args.gating_d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                ii)
            
            setting = setting + '_tsfn-experts' + tsfn_experts_str + '_tsft-experts' + tsft_experts_str
            if os.path.exists('./{}_gating_results/'.format("test") + setting + '/'):
                print("This setting {} has been trained and tested, skip it".format(setting))
                continue


            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            now_mse=exp.test(setting, test=1)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_agg{}_eip{}_pl{}_dm{}_nh{}_el{}_df{}_{}'.format(
                args.domain,
                args.task_name,
                args.agg_type,
                args.expert_input_type,
                args.pred_len,
                args.gating_d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                ii)
            
        setting = setting + '_tsfn-experts' + tsfn_experts_str + '_tsft-experts' + tsft_experts_str


        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        now_mse=exp.test(setting, test=1)

        torch.cuda.empty_cache()
    

import argparse
import numpy as np
from os.path import join, exists
import pandas as pd
import os
import torch
from os.path import join, exists

def extract_expert_features(dataset_folder, exp_template, expert_type, freq, 
                    domain, expert, pl, results, verbose=False):
    
    model_folder_name = exp_template.format(expert_type, domain, pl, expert, pl)
    model_full_path = join(dataset_folder, model_folder_name)
    
    if verbose:
        print("Extracting configuration for expert {}".format(model_folder_name))
    if (exists(model_full_path)):
        try:
            if expert_type == "tsfn":
                latent_num_embs_path = join(model_full_path,'latent_num_embs.pth')
                h = torch.stack(torch.load(latent_num_embs_path, map_location="cpu"))
                latent_dim = h.reshape(h.shape[0]*h.shape[1], -1).shape[-1]
            elif expert_type == "tsft":
                # Load the latent text embeddings
                latent_txt_embs_path = join(model_full_path,'latent_text_embs.pth')
                h = torch.load(latent_txt_embs_path, map_location="cpu")
                h = torch.stack([torch.nn.functional.adaptive_avg_pool1d(h_batch.transpose(1,2), 1).squeeze(2) for h_batch in h])
                h = h.reshape(h.shape[0]*h.shape[1], -1)
                latent_dim = h.shape[-1]
                #h = torch.stack(torch.load(latent_txt_embs_path, map_location="cpu"))
                #latent_dim = h.reshape(h.shape[0]*h.shape[1], -1).shape[-1]
            else:
                raise ValueError("expert type not found")
        except Exception as e:
            print("Failed extracting configuration for expert {}".format(model_folder_name))
            print("Error loading latent embeddings: {}".format(e))
            return
        if verbose:
            print("type: {} freq: {} domain: {} expert: {} pl: {} latent dim: {}".format(expert_type, 
                                                                    freq, domain, expert, pl, latent_dim))
        results["domain"].append(domain)
        results["freq"].append(freq)
        results["pl"].append(pl)
        results["expert"].append(expert)
        results["expert_type"].append(expert_type)
        results["folder_path"].append(model_folder_name)
        results["latent_dim"].append(latent_dim)
    else:
        print("Failed extracting configuration for expert {}".format(model_folder_name))
        print("folder not found!")
    if verbose:
        print("-------------------------------------------------------")

def parse_folders(exp_config, exp_template, dataset_folder, results, freq):
    """
    Parse the folders in the results folder and extract the metrics for each model.
    """
    for domain in exp_config["domain"]:
        for pl in exp_config["pl"]:
            for expert in exp_config["tsfn_experts"]:
                expert_type = "tsfn"
                extract_expert_features(dataset_folder, exp_template, expert_type, 
                                      freq, domain, expert, pl, results)
            for expert in exp_config["tsft_experts"]:
                expert_type = "tsft"
                extract_expert_features(dataset_folder, exp_template, expert_type, 
                                      freq, domain, expert, pl, results)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract expert configurations')
    parser.add_argument('--dataset_folder', type=str, default='train_gating_dataset',
                        help='path to dataset folder')
    parser.add_argument('--out_file', type=str, default='all_experts_config.csv',
                        help='output file name')

    args = parser.parse_args()
    print("extracting expert configuration for gating dataset")
    
    all_experts_config = {"domain":[], 
                "pl":[],
                "expert":[],
                "freq":[],
                 "latent_dim":[], "expert_type":[], "folder_path":[]}
    
    # Daily
    
    exp_config = {"domain":["Environment"], "tsft_experts":["LLAMA2","GPT2","BERT"],
                     "tsfn_experts":["Informer", "Reformer", "DLinear", "PatchTST", "FiLM"], "pl":[48, 96, 192, 336]}
    
    exp_template = "long_term_forecast_{}_{}_2021_24_{}_fullLLM_0_{}_custom_ftS_sl96_ll48_pl{}_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"

    freq = 'daily'
    parse_folders(exp_config, exp_template, args.dataset_folder, all_experts_config, freq)
    

    
    # Weekly
    exp_config = {"domain":["Energy","Public_Health"], "tsft_experts":["LLAMA2","GPT2","BERT"],
                     "tsfn_experts":["Informer", "Reformer", "DLinear", "PatchTST", "FiLM"], "pl":[12, 24, 36, 48]}
    exp_template = "long_term_forecast_{}_{}_2021_24_{}_fullLLM_0_{}_custom_ftS_sl36_ll18_pl{}_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"
    freq = 'weekly'
    parse_folders(exp_config, exp_template, args.dataset_folder, all_experts_config, freq)

     # Monthly
    exp_config = {"domain":["Algriculture","Climate","Economy","Security","SocialGood","Traffic"],
                   "tsft_experts":["LLAMA2","GPT2","BERT"],
                     "tsfn_experts":["Informer", "Reformer", "DLinear", "PatchTST", "FiLM"], "pl":[6, 8, 10, 12]}
    
    exp_template = "long_term_forecast_{}_{}_2021_24_{}_fullLLM_0_{}_custom_ftS_sl8_ll4_pl{}_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"
    freq = 'monthly'
    parse_folders(exp_config, exp_template, args.dataset_folder, all_experts_config, freq)

    # Save the results to a file
    df = pd.DataFrame(all_experts_config)
    df.to_csv(args.out_file, index=False)
    print("Results saved to {}".format(args.out_file))
    print("-------------------------------------------------------")
    

                
                    
                

  

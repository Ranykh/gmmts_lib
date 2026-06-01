tsfn_models=("Informer" "Reformer" "DLinear" "PatchTST" "FiLM")
tsft_models=("GPT2" "BERT" "LLAMA2")
agg_types=("direct")
input_types=("latent")
seeds=(2021)

for ((i=0; i<2; i++))
do
 if [ $i == 1 ]
  then
    domains=("Energy" "Public_Health")
    pred_lengths=(12 24 36 48)
  fi
if [ $i == 0 ]
  then
    #domains=("Algriculture" "Climate" "Security" "Traffic" "Security" "SocialGood" "Economy")
    domains=("Security")
    pred_lengths=(6 8 10 12)
    fi
    for seed in "${seeds[@]}"
    do
      for tsfn_model_name in "${tsfn_models[@]}"
      do
        for tsft_model_name in "${tsft_models[@]}"
        do
          for agg_type in "${agg_types[@]}"
          do
              for domain in "${domains[@]}"
              do
                for pred_len in "${pred_lengths[@]}"
                do
                for eip in "${input_types[@]}"
                do
                    echo "Running domain $domain with $tsfn_model_name - $tsft_model_name pred_len $pred_len agg $agg_type input type $input_type seed $seed"
                    python -u run_offline_gating.py \
                --task_name mm_long_term_forecast \
                --pred_len $pred_len \
                --is_training 1 \
                --agg_type $agg_type \
                --seed $seed \
                --expert_input_type $eip \
                --tsfn_experts $tsfn_model_name \
                --tsft_experts $tsft_model_name \
                --domain $domain
                    done 
                done
              done
          done
        done
      done
    done
done

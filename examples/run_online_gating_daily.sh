export CUDA_VISIBLE_DEVICES=$1

# Check if MM_TSFLIB_PATH is set
if [ -z "$MM_TSFLIB_PATH" ]; then
    echo "Error: MM_TSFLIB_PATH environment variable is not set"
    echo "Please set it to the path of MM-TSFlib repository"
    exit 1
fi

echo "Using MM-TSFlib from: $MM_TSFLIB_PATH"

all_models=("Informer" "Reformer" "DLinear" "PatchTST" "FiLM")
llm_models=("GPT2" "BERT")  # Removed LLAMA2 as it requires special setup

root_paths=("${MM_TSFLIB_PATH}/data/Environment")
data_paths=("NewYork_AQI_Day.csv") 
pred_lengths=(48 96 192 336)
seeds=(2021)
use_fullmodel=0
agg_type="direct"
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for model_name in "${all_models[@]}"
  do
  for llm_model in "${llm_models[@]}"
  do
    for ((i=0; i<$length; i++))
    do
      for pred_len in "${pred_lengths[@]}"
      do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        model_id=$(basename ${root_path})

        echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
        python -u run_online_gating.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path \
          --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --des 'Exp' \
          --seed $seed \
          --type_tag "#F#" \
          --text_len 4 \
          --pool_type "avg" \
          --save_name "results_weekly" \
          --llm_model $llm_model \
          --huggingface_token 'NA'\
          --use_fullmodel $use_fullmodel \
          --agg_type $agg_type
      done
    done
  done
done
done


#!/bin/bash

# GMM-TS: Run online gating experiments on weekly datasets
# Usage: bash examples/run_online_gating_weekly.sh GPU_ID

export CUDA_VISIBLE_DEVICES=$1

export MM_TSFLIB_PATH=${MM_TSFLIB_PATH:-"../MM-TSFlib"}

if [ ! -d "$MM_TSFLIB_PATH" ]; then
    echo "Error: MM-TSFlib not found at $MM_TSFLIB_PATH"
    echo "Please set MM_TSFLIB_PATH environment variable or adjust the path in this script"
    exit 1
fi

echo "Using MM-TSFlib from: $MM_TSFLIB_PATH"

all_models=("Informer" "Reformer" "DLinear" "PatchTST" "FiLM")
llm_models=("GPT2" "BERT")

root_paths=(
    "$MM_TSFLIB_PATH/data/Energy"
    "$MM_TSFLIB_PATH/data/Public_Health"
)
data_paths=("US_GasolinePrice_Week.csv" "US_FLURATIO_Week.csv")
pred_lengths=(12 24 36 48)
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
          --seq_len 36 \
          --label_len 18 \
          --pred_len $pred_len \
          --des 'Exp' \
          --seed $seed \
          --type_tag "#F#" \
          --text_len 4 \
          --pool_type "avg" \
          --save_name "results_weekly" \
          --llm_model $llm_model \
          --huggingface_token "${HUGGINGFACE_TOKEN:-}" \
          --use_fullmodel $use_fullmodel \
          --agg_type $agg_type
      done
    done
  done
done
done

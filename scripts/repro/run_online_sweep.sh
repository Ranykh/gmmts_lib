#!/usr/bin/env bash
# Online (joint) GMM-TS sweep over all horizons of ONE domain, following the paper's
# forecasting setup (examples/run_online_gating_*.sh). One domain per GPU.
#
# Usage (pin the card with CUDA_VISIBLE_DEVICES; the python side then always uses --gpu 0):
#   CUDA_VISIBLE_DEVICES=2 bash scripts/repro/run_online_sweep.sh TAG AGG DOMAIN "SEEDS" [extra args]
#
#   TAG     label for this arm, e.g. table1 or mogu  (becomes a folder under RUNS_DIR)
#   AGG     direct | latent | hierarchical   (learned GatingNet, vanilla GMM-TS)
#           mogu                             (inverse-variance gate over all experts)
#   DOMAIN  Environment Energy Public_Health Algriculture Climate Economy Security SocialGood Traffic
#   SEEDS   e.g. "2021" or "2021 2022 2023"
#   extra   passed through to run_online_gating.py
#
# Env overrides: TSFN (default PatchTST; comma list for >1 numeric expert), LLM (default GPT2),
#                RUNS_DIR (default ./runs), EPOCHS (default 10, as in the paper).
#
# Every (TAG, seed, domain) runs in its own working directory. run_online_gating.py writes
# results/checkpoints relative to the CWD and its `setting` string does not contain the seed,
# so without this a second seed would hit the "already trained" skip-guard and silently
# report the first seed's numbers.
set -euo pipefail

if [ $# -lt 4 ]; then
  sed -n '2,21p' "$0"; exit 1
fi
TAG=$1; AGG=$2; DOMAIN=$3; SEEDS=$4; shift 4
EXTRA=("$@")

: "${MM_TSFLIB_PATH:?export MM_TSFLIB_PATH=/path/to/MM-TSFlib (pinned at e789ce78)}"
TSFN=${TSFN:-PatchTST}
LLM=${LLM:-GPT2}
EPOCHS=${EPOCHS:-10}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNS_DIR=$(mkdir -p "${RUNS_DIR:-$PWD/runs}" && cd "${RUNS_DIR:-$PWD/runs}" && pwd)
CONFIG="$REPO/all_experts_config.csv"
# the learned gate sizes its projections from this csv; the MoGU gate does not read it
if [ "$AGG" != "mogu" ] && [ ! -f "$CONFIG" ]; then
  echo "missing $CONFIG -- run scripts/repro/make_online_experts_config.py first"; exit 1
fi

case $DOMAIN in
  Environment)    DATA=NewYork_AQI_Day.csv;                      SL=96; LL=48; PLS="48 96 192 336" ;;
  Energy)         DATA=US_GasolinePrice_Week.csv;                SL=36; LL=18; PLS="12 24 36 48" ;;
  Public_Health)  DATA=US_FLURATIO_Week.csv;                     SL=36; LL=18; PLS="12 24 36 48" ;;
  Algriculture)   DATA=US_RetailBroilerComposite_Month.csv;      SL=8;  LL=4;  PLS="6 8 10 12" ;;
  Climate)        DATA=US_precipitation_month.csv;               SL=8;  LL=4;  PLS="6 8 10 12" ;;
  Economy)        DATA=US_TradeBalance_Month.csv;                SL=8;  LL=4;  PLS="6 8 10 12" ;;
  Security)       DATA=US_FEMAGrant_Month.csv;                   SL=8;  LL=4;  PLS="6 8 10 12" ;;
  SocialGood)     DATA=Unadj_UnemploymentRate_ALL_processed.csv; SL=8;  LL=4;  PLS="6 8 10 12" ;;
  Traffic)        DATA=US_VMT_Month.csv;                         SL=8;  LL=4;  PLS="6 8 10 12" ;;
  *) echo "unknown domain: $DOMAIN"; exit 1 ;;
esac
[ -f "$MM_TSFLIB_PATH/data/$DOMAIN/$DATA" ] || { echo "missing $MM_TSFLIB_PATH/data/$DOMAIN/$DATA"; exit 1; }

for SEED in $SEEDS; do
  WORK="$RUNS_DIR/$TAG/seed$SEED/$DOMAIN"
  mkdir -p "$WORK"
  for PL in $PLS; do
    echo "=== $TAG | $DOMAIN | pl=$PL | seed=$SEED | $TSFN + $LLM | agg=$AGG ==="
    (
      cd "$WORK"
      python -u "$REPO/run_online_gating.py" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$MM_TSFLIB_PATH/data/$DOMAIN" \
        --data_path "$DATA" \
        --model_id "${DOMAIN}_${SEED}_${SL}_${PL}_${LLM}" \
        --model "$TSFN" \
        --data custom \
        --features M \
        --seq_len "$SL" \
        --label_len "$LL" \
        --pred_len "$PL" \
        --des Exp \
        --seed "$SEED" \
        --type_tag "#F#" \
        --text_len 4 \
        --pool_type avg \
        --llm_model "$LLM" \
        --use_fullmodel 0 \
        --agg_type "$AGG" \
        --train_epochs "$EPOCHS" \
        --all_experts_config "$CONFIG" \
        --save_name "results_${TAG}.txt" \
        --gpu 0 \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "log_pl${PL}.txt"
    )
  done
done
echo "done: $RUNS_DIR/$TAG (collect with scripts/repro/collect_online_results.py)"

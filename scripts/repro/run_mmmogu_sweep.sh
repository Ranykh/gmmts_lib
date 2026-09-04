#!/usr/bin/env bash
# =============================================================================
# MM-MoGU vs GMM-TS sweep -- ONLINE (joint) gating
#
#   bash scripts/repro/run_mmmogu_sweep.sh <gpu_id> <what>
#
#     what = smoke     one short run, proves the whole cross-repo pipeline
#            g3        GMM-TS baseline reproduced in OUR harness (agg_type=direct)
#            g4        MM-MoGU, inverse-variance gate (agg_type=inv_var, prob_expert=1)
#            g4norm    MM-MoGU with per-modality log-variance normalisation
#            all       g3 then g4
#
#   Examples
#     bash scripts/repro/run_mmmogu_sweep.sh 2 smoke
#     bash scripts/repro/run_mmmogu_sweep.sh 2 all
#     SKIP_DONE=1 bash scripts/repro/run_mmmogu_sweep.sh 2 g4     # resume
#     DOMAINS="Economy" bash scripts/repro/run_mmmogu_sweep.sh 2 g4
#
# -----------------------------------------------------------------------------
# THE TRAINING RECIPE -- why this is the ONLINE script and never the offline one
# -----------------------------------------------------------------------------
# The experts must be trained JOINTLY with the uncertainty estimator and the
# final decision after the gate. exp_online_gating_long_term_forecasting.py
# already does exactly that, and it is worth knowing where, because the offline
# script does something different and looks superficially similar:
#
#   TS experts       _select_optimizer()        Adam over m.parameters() for each
#                                               expert -- FULL backbone + its UncHead
#   LLM backbone     line ~350                  requires_grad = False   (FROZEN)
#   LLM projection   _select_optimizer_proj()   mlp_proj                (TRAINED)
#   LLM uncertainty  _select_optimizer_mlp()    mlp + text_unc_head     (TRAINED)
#   Gate             _select_optimizer_gating() gating_module           (TRAINED)
#
# All four step per batch. run_offline_gating.py is the pretrained/frozen-expert
# path -- do NOT use it for these results.
#
# -----------------------------------------------------------------------------
# THE OVERWRITE TRAP -- read before changing how runs are named
# -----------------------------------------------------------------------------
# run_online_gating.py builds `setting` (line 210) as
#
#   {domain}_{task_name}_agg{agg_type}_eip{...}_pl{...}_dm{...}_nh{...}
#   _el{...}_df{...}_{itr}_tsfn-experts{model}_tsft-experts{llm_model}
#
# and `domain` is NOT an argument -- line 152 derives it from the last path
# segment of --root_path. So the folder name does NOT encode:
#
#     --seed          --prob_expert          --inv_var_norm          --text_len
#
# Two consequences, both silent:
#   1. inv_var_norm=none and inv_var_norm=per_modality write to the SAME folder.
#   2. run_online_gating.py line 223 SKIPS a setting whose folder already exists.
#      So the second config is not overwritten -- it is never run at all, and the
#      log says "has been trained and tested, skip it" as if that were fine.
#
# --model_id does not help: it is not in the setting string at all, despite the
# example scripts carefully constructing one.
#
# The fix used here, without modifying the repo: after each run completes, the
# results folder is RENAMED to carry the missing fields:
#
#     <setting>__norm-<inv_var_norm>__pe<prob_expert>__seed<seed>
#
# That preserves every run distinctly AND clears the way for the next config,
# because the skip guard no longer finds the original name. Our own SKIP_DONE
# checks the renamed folder instead.
#
# -----------------------------------------------------------------------------
# WHAT IS NOT SAVED
# -----------------------------------------------------------------------------
# The online exp writes metrics.npy / pred.npy / true.npy only. It does NOT save
# the gate weights, even though GatingNet.forward(data, return_w=True) returns
# them for every aggregation type including inv_var. So there is no gate-collapse
# evidence in these runs: a gate that has settled on a constant blend will look
# identical to one that is routing. collect_gmmts.py reports this gap rather than
# pretending the column is empty for a benign reason. Adding the save is a small
# flag-guarded change to the exp file -- deliberately NOT bundled here, so this
# sweep runs against the repo exactly as committed.
# =============================================================================
set -uo pipefail

GPU_ID=${1:?usage: run_mmmogu_sweep.sh <gpu_id> <smoke|g3|g4|g4norm|all>}
WHAT=${2:-smoke}

export CUDA_VISIBLE_DEVICES="$GPU_ID"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# gmmts_lib vendors no models or data -- it imports both from MM-TSFlib.
export MM_TSFLIB_PATH="${MM_TSFLIB_PATH:-$HOME/msc/MM-TSFlib}"
if [ ! -d "$MM_TSFLIB_PATH" ]; then
  echo "ERROR: MM-TSFlib not found at $MM_TSFLIB_PATH" >&2
  echo "       clone it and/or set MM_TSFLIB_PATH." >&2
  exit 1
fi

# ----- the experiment grid --------------------------------------------------
# Three domains, chosen so the gate has something to do and the runs are cheap.
# From the GMM-TS ablation (section 6.1), MSE degradation when the dynamic gate
# is removed:  Economy 12.728  |  Traffic 0.927  |  Social Good 0.311.
# Economy is the strongest case in the whole ablation; Social Good is the weak
# case, included deliberately so the selection is not cherry-picked. All three
# are Monthly, so one protocol covers them. Scales span 0.04 to 1.09 -- never
# average across them.
DOMAINS="${DOMAINS:-Economy Traffic SocialGood}"

declare -A DATA_FILE=(
  [Economy]="US_TradeBalance_Month.csv"
  [Traffic]="US_VMT_Month.csv"
  [SocialGood]="Unadj_UnemploymentRate_ALL_processed.csv"
  [Security]="US_FEMAGrant_Month.csv"          # stretch: the sharpest test, text
  [Algriculture]="US_RetailBroilerComposite_Month.csv"   # hurts here, so a good
  [Climate]="US_precipitation_month.csv"                 # gate should suppress it
)

# Three expert pairs, varying ONE factor at a time from a common corner, so a
# difference can be attributed. "DLinear x GPT2" is the anchor: DLinear is the
# strongest numeric expert in the published ablation (Table 20) and GPT2 is the
# cheapest text expert.
#   pair 2 changes the numeric backbone only
#   pair 3 changes the text expert only
# GPT3.5 is deliberately absent: it needs the Final_Output column, which exists
# in Energy, Public_Health and Traffic only -- Economy and SocialGood KeyError.
PAIRS="${PAIRS:-DLinear:GPT2 PatchTST:GPT2 DLinear:LLAMA2}"

HORIZONS="${HORIZONS:-6 8 10 12}"
SEEDS="${SEEDS:-2021}"

# ----- protocol constants, from examples/run_online_gating_monthly.sh --------
SEQ_LEN=8
LABEL_LEN=4
TEXT_LEN=4          # run_online_gating.py defaults to 3, but the columns are
                    # Final_Search_{2,4,6} -- 3 is a KeyError.
POOL_TYPE="avg"
TYPE_TAG="#F#"
USE_FULLMODEL=0
EXPERT_INPUT_TYPE="${EXPERT_INPUT_TYPE:-prediction}"

RESULTS_DIR="$REPO_ROOT/test_online_gating_results"
LOG_DIR="${LOG_DIR:-$HOME/msc/logs/gmmts/$WHAT}"
SKIP_DONE="${SKIP_DONE:-0}"
mkdir -p "$LOG_DIR"

# exp_online_gating_long_term_forecasting.py line 394 reads this file
# unconditionally in __init__, before a single batch is loaded, and it is not in
# the repository. Fail here with something readable rather than a bare
# FileNotFoundError forty lines into a traceback.
ALL_EXPERTS_CONFIG="${ALL_EXPERTS_CONFIG:-$REPO_ROOT/all_experts_config.csv}"
if [ ! -f "$ALL_EXPERTS_CONFIG" ]; then
  echo "ERROR: $ALL_EXPERTS_CONFIG not found." >&2
  echo "       Online gating cannot start without it. Generate one with:" >&2
  echo "         python scripts/repro/make_experts_config.py --domains $DOMAINS" >&2
  exit 1
fi

# ----- provenance -----------------------------------------------------------
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "NO-GIT")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | head -c1)
MM_SHA=$(git -C "$MM_TSFLIB_PATH" rev-parse HEAD 2>/dev/null || echo "NO-GIT")
MM_BRANCH=$(git -C "$MM_TSFLIB_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
RUN_STAMP=$(date +%F_%H%M%S)
MANIFEST="$LOG_DIR/manifest_${RUN_STAMP}.txt"

{
  echo "what:            $WHAT"
  echo "started:         $(date -Is)"
  echo "gpu:             physical card $GPU_ID"
  echo "gmmts_sha:       $GIT_SHA"
  echo "gmmts_dirty:     $([ -n "$GIT_DIRTY" ] && echo YES || echo no)"
  echo "mm_tsflib_path:  $MM_TSFLIB_PATH"
  echo "mm_tsflib_sha:   $MM_SHA  (branch $MM_BRANCH)"
  echo "domains:         $DOMAINS"
  echo "pairs:           $PAIRS"
  echo "horizons:        $HORIZONS"
  echo "seeds:           $SEEDS"
} | tee "$MANIFEST"

# prob_expert=1 loads the mm-mogu experts, which is where sigma^2 comes from.
# Any branch without them cannot produce an inverse-variance gate at all.
if [ "$WHAT" != "g3" ] && [ "$MM_BRANCH" != "mm-mogu" ]; then
  echo ""
  echo "!! MM-TSFlib is on branch '$MM_BRANCH', not 'mm-mogu'."
  echo "!! prob_expert=1 needs the mm-mogu experts (they return (pred, sigma^2))."
  echo "!! g4 will fail without them. g3 is fine on any branch."
fi
if [ -n "$GIT_DIRTY" ]; then
  echo ""
  echo "!! gmmts_lib working tree is DIRTY -- results will not be tied to $GIT_SHA."
fi

N_OK=0; N_FAIL=0; N_SKIP=0; FAILED_RUNS=()

# =============================================================================
# run_one <domain> <tsf_n> <tsf_t> <pred_len> <seed> <agg_type> <prob_expert> <inv_var_norm>
# =============================================================================
run_one () {
  local domain=$1 tsfn=$2 tsft=$3 pl=$4 seed=$5 agg=$6 pe=$7 norm=$8
  local rc elapsed start mse

  local dfile=${DATA_FILE[$domain]:-}
  if [ -z "$dfile" ]; then
    echo "!! no data file mapped for domain '$domain', skipping" >&2
    return 1
  fi
  local root="$MM_TSFLIB_PATH/data/$domain"

  # The setting string the repo will build, reconstructed here so we can rename
  # the folder afterwards. Keep in sync with run_online_gating.py line 210.
  local setting="${domain}_long_term_forecast_agg${agg}_eip${EXPERT_INPUT_TYPE}_pl${pl}"
  setting="${setting}_dm256_nh8_el2_df2048_0_tsfn-experts${tsfn}_tsft-experts${tsft}"
  local final="${setting}__norm-${norm}__pe${pe}__seed${seed}"

  local tag="${domain}_${tsfn}_${tsft}_pl${pl}_agg${agg}_norm${norm}_pe${pe}_s${seed}"
  local log="$LOG_DIR/${tag}.log"

  if [ "$SKIP_DONE" = "1" ] && [ -d "$RESULTS_DIR/$final" ]; then
    echo "--- SKIP (already done): $tag"
    N_SKIP=$((N_SKIP + 1)); return 0
  fi

  # Clear a stale un-renamed folder, or the repo's own skip guard fires and the
  # run silently does nothing.
  if [ -d "$RESULTS_DIR/$setting" ]; then
    echo "    (removing stale un-renamed folder from an earlier attempt)"
    rm -rf "$RESULTS_DIR/$setting"
  fi

  echo ""
  echo "=== $tag"
  echo "    log -> $log"
  start=$(date +%s)

  {
    python -u run_online_gating.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root" \
      --data_path "$dfile" \
      --model_id "${domain}_${tsfn}_${tsft}_pl${pl}_${agg}_${norm}_s${seed}" \
      --model "$tsfn" \
      --llm_model "$tsft" \
      --data custom \
      --features M \
      --seq_len $SEQ_LEN \
      --label_len $LABEL_LEN \
      --pred_len "$pl" \
      --des 'Exp' \
      --seed "$seed" \
      --type_tag "$TYPE_TAG" \
      --text_len $TEXT_LEN \
      --pool_type "$POOL_TYPE" \
      --save_name "results_${WHAT}" \
      --huggingface_token "${HUGGINGFACE_TOKEN:-}" \
      --use_fullmodel $USE_FULLMODEL \
      --expert_input_type "$EXPERT_INPUT_TYPE" \
      --agg_type "$agg" \
      --prob_expert "$pe" \
      --inv_var_norm "$norm" \
      --all_experts_config "$ALL_EXPERTS_CONFIG"
  } 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - start ))

  if [ "$rc" -eq 0 ] && [ -f "$RESULTS_DIR/$setting/metrics.npy" ]; then
    mv "$RESULTS_DIR/$setting" "$RESULTS_DIR/$final"
    mse=$(grep "^mse:" "$log" | tail -1)
    echo "    OK  (${elapsed}s)  $mse"
    echo "    -> $final"
    echo "$tag | ${elapsed}s | $mse" >> "$MANIFEST"
    N_OK=$((N_OK + 1))
  elif grep -q "has been trained and tested, skip it" "$log"; then
    # The repo's guard fired. Almost always a genuine collision, not a real skip.
    echo "    !! the repo SKIPPED this run -- a folder with the same setting existed."
    echo "    !! that means two configs collide. Check the rename logic above."
    echo "$tag | ${elapsed}s | REPO-SKIPPED (collision)" >> "$MANIFEST"
    FAILED_RUNS+=("$tag (repo-skipped: setting collision)")
    N_FAIL=$((N_FAIL + 1))
  else
    echo "    FAILED (rc=$rc, ${elapsed}s) -- see $log"
    echo "$tag | ${elapsed}s | FAILED rc=$rc" >> "$MANIFEST"
    FAILED_RUNS+=("$tag")
    N_FAIL=$((N_FAIL + 1))
  fi
}

sweep () {   # sweep <agg_type> <prob_expert> <inv_var_norm>
  local agg=$1 pe=$2 norm=$3
  for seed in $SEEDS; do
    for domain in $DOMAINS; do
      for pair in $PAIRS; do
        local tsfn="${pair%%:*}" tsft="${pair##*:}"
        for pl in $HORIZONS; do
          run_one "$domain" "$tsfn" "$tsft" "$pl" "$seed" "$agg" "$pe" "$norm"
        done
      done
    done
  done
}

# =============================================================================
case "$WHAT" in

  smoke)
    # One cheap run on the smallest grid cell. Proves: MM_TSFLIB_PATH resolves,
    # the data loads, the frozen LLM downloads and runs, joint training steps,
    # and a metrics.npy lands. ~ a few minutes.
    echo "### smoke: Economy, DLinear x GPT2, h=6, agg=direct ###"
    HORIZONS=6 DOMAINS=Economy PAIRS=DLinear:GPT2 sweep direct 0 none
    echo ""
    echo "### expect: test_online_gating_results/Economy_..._aggdirect_..._pl6_"
    echo "###         ..._tsfn-expertsDLinear_tsft-expertsGPT2__norm-none__pe0__seed2021/"
    echo "###         containing metrics.npy"
    ;;

  g3)
    # GMM-TS reproduced in OUR harness. This is the correctness gate: our
    # agg_type=direct must land near the published G3 ATTN column of
    # Pairwise_Baselines. If it does, the harness is trustworthy and the G4-vs-G3
    # comparison below is apples-to-apples in one codebase, which is a far
    # stronger claim than comparing our numbers against someone else's printout.
    # Direct aggregation is the paper's safe default (Table 17): best on Economy
    # and Traffic, and latent is catastrophic on Economy (1.29 vs 0.22).
    echo "### G3: GMM-TS baseline, agg_type=direct ###"
    sweep direct 0 none
    ;;

  g4)
    # MM-MoGU: the inverse-variance gate. prob_expert=1 makes the experts emit
    # (pred, sigma^2); the gate weights each expert by 1/sigma^2 with zero
    # trained gating parameters.
    echo "### G4: MM-MoGU, agg_type=inv_var, prob_expert=1, norm=none ###"
    sweep inv_var 1 none
    ;;

  g4norm)
    # Per-modality log-variance normalisation. Known from the earlier controlled
    # bake-off to be PROVABLY invariant to per-expert variance rescaling
    # (z(log c*sigma^2) = z(log sigma^2)), so it cannot benefit from calibration.
    # Run it as the contrast, not as the expected winner.
    echo "### G4-norm: agg_type=inv_var, prob_expert=1, norm=per_modality ###"
    sweep inv_var 1 per_modality
    ;;

  all)
    echo "### G3 then G4 ###"
    sweep direct 0 none
    sweep inv_var 1 none
    ;;

  *)
    echo "unknown '$WHAT'. use: smoke | g3 | g4 | g4norm | all" >&2
    exit 1
    ;;
esac

# =============================================================================
echo ""
echo "============================================================"
echo "$WHAT finished at $(date -Is)"
echo "  ok: $N_OK   skipped: $N_SKIP   failed: $N_FAIL"
if [ "$N_FAIL" -gt 0 ]; then
  echo ""
  for f in "${FAILED_RUNS[@]}"; do echo "    - $f"; done
  echo ""
  echo "  re-run with SKIP_DONE=1 to leave the good ones alone."
fi
echo ""
echo "  manifest:  $MANIFEST"
echo "  gmmts_sha: $GIT_SHA"
echo "  mm_sha:    $MM_SHA"
echo ""
echo "collect with:"
echo "  python scripts/repro/collect_gmmts.py --out ~/msc/logs/gmmts_${WHAT}.csv"
echo "============================================================"

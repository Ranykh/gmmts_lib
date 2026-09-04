#!/usr/bin/env python3
"""Generate the all_experts_config.csv that online gating refuses to start without.

    python scripts/repro/make_experts_config.py                       # the 7-Sept grid
    python scripts/repro/make_experts_config.py --domains Economy Traffic SocialGood
    python scripts/repro/make_experts_config.py --all-monthly --out all_experts_config.csv

WHY THIS EXISTS
---------------
exp_online_gating_long_term_forecasting.py line 394 does, unconditionally, in
__init__:

    all_expert_config_df = pd.read_csv(args.all_experts_config)

and `all_experts_config.csv` is NOT in the repository. Without it, every online
gating run dies with FileNotFoundError before a single batch is loaded.

The repo's own generator, scripts/dataset_prep_scripts/prepare_all_expert_config.py,
cannot help on a fresh clone: it reads `latent_num_embs.pth` / `latent_text_embs.pth`
out of a `train_gating_dataset/` folder that only exists after you have run the
save_tsfns_/save_tsfts_latents_predictions_*.sh scripts over every expert, every
domain and every horizon. That is a large prerequisite stage, and it is not on
the critical path for the comparison we actually need.

WHAT THE FILE IS ACTUALLY USED FOR
----------------------------------
Only six fields are read (lines 398-407), and only one of them ever affects the
computation:

    latent_dim    sizes nn.Linear(expert_dim, gating_d_model) per expert,
                  Gating.py line 42 -- BUT ONLY when expert_input_type == "latent".
    domain, freq, pl, expert_type    carried into the config dict, not used to
                  build anything.
    folder_path   used by the OFFLINE path to load pretrained experts. The online
                  path builds experts fresh in _build_model(), so it is inert here.

Three consequences worth knowing:

  * agg_type=inv_var is a separate, latent-free path (Gating.py line 175), so the
    expert projections are not built at all -- latent_dim is never read there.
  * With --expert_input_type prediction, Gating.py takes expert_dim = pred_len and
    also never reads latent_dim.
  * On the mm-mogu branch, expert_input_type=latent and =prediction are
    NUMERICALLY IDENTICAL for numeric experts, because none of those experts
    returns a real latent -- see the note on TSFN_LATENT_IS_PRED_LEN below.

The text-expert dimensions below are architectural facts, not guesses:
GPT2/GPT2M/GPT2L/GPT2XL and BERT-base hidden sizes, and LLaMA-2-7B's 4096.

NOT EVERY EXPERT CAN DO INVERSE-VARIANCE GATING
-----------------------------------------------
Only PatchTST and iTransformer were given an UncHead on mm-mogu. DLinear, FiLM,
Informer and Reformer are unmodified from upstream and return a plain tensor, so
prob_expert=1 crashes with "NoneType object is not subscriptable". This file will
happily emit rows for them -- they are valid for agg_type=direct -- but an
inverse-variance run must use PatchTST or iTransformer.
"""
import argparse
import csv
import sys

# Hidden sizes. These are the real model dimensions, so they are correct whenever
# the text latent is the pooled last hidden state.
TSFT_DIM = {
    "BERT": 768,
    "GPT2": 768, "GPT2M": 1024, "GPT2L": 1280, "GPT2XL": 1600,
    "LLAMA2": 4096, "LLAMA3": 4096,
    "Doc2Vec": 100, "ClosedLLM": 768,
}

# Numeric-expert latent width. On the mm-mogu branch this is NOT d_model, and
# that distinction was verified against the source rather than assumed:
#
#   * PatchTST/iTransformer with prob_expert=1 return (pred, sigma^2).
#   * Every expert with prob_expert=0 returns a PLAIN TENSOR.
#   * No expert on that branch returns (pred, latent).
#
# So exp_online_gating_long_term_forecasting._unpack_expert_result always falls
# through to `m_latents = m_result.reshape(B, -1)` -- the "latent" IS the
# flattened prediction, of width pred_len * c_out. run.py forces features='S',
# so c_out = 1 and the width is exactly pred_len.
#
# Consequence: for numeric experts, expert_input_type=latent and
# expert_input_type=prediction are numerically IDENTICAL on this branch.
# Setting this to a fixed d_model (512) would build nn.Linear(512, gating_d_model)
# and feed it a pred_len-wide vector -- a shape mismatch.
TSFN_LATENT_IS_PRED_LEN = True

TSFN_EXPERTS = ["Informer", "Reformer", "DLinear", "PatchTST", "FiLM"]

# Frequency and horizon grid, from the repo's own prepare_all_expert_config.py.
FREQ_GROUPS = {
    "monthly": (["Algriculture", "Climate", "Economy",
                 "Security", "SocialGood", "Traffic"], [6, 8, 10, 12], 8, 4),
    "weekly": (["Energy", "Public_Health"], [12, 24, 36, 48], 36, 18),
    "daily": (["Environment"], [48, 96, 192, 336], 96, 48),
}

# The folder-name shape the offline path expects. Written for completeness; the
# online path never opens it.
FOLDER_TEMPLATE = ("long_term_forecast_{expert_type}_{domain}_2021_24_{pl}_fullLLM_0_"
                   "{expert}_custom_ftS_sl{sl}_ll{ll}_pl{pl}_dm512_nh8_el2_dl1_df2048_"
                   "expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0")

FIELDS = ["domain", "pl", "expert", "freq", "latent_dim", "expert_type", "folder_path"]


def freq_of(domain):
    for freq, (domains, _pls, _sl, _ll) in FREQ_GROUPS.items():
        if domain in domains:
            return freq
    return None


def build(domains, horizons, tsfn, tsft, tsfn_latent):
    rows, skipped = [], []
    for domain in domains:
        freq = freq_of(domain)
        if freq is None:
            skipped.append(f"{domain}: unknown domain, not in any frequency group")
            continue
        default_pls, sl, ll = FREQ_GROUPS[freq][1], FREQ_GROUPS[freq][2], FREQ_GROUPS[freq][3]
        pls = horizons or default_pls
        for pl in pls:
            if pl not in default_pls:
                skipped.append(f"{domain} h={pl}: not a published horizon for "
                               f"{freq} domains {default_pls}")
            for expert in tsfn:
                # pred_len, not d_model -- see the note on TSFN_LATENT_IS_PRED_LEN.
                rows.append({
                    "domain": domain, "pl": pl, "expert": expert, "freq": freq,
                    "latent_dim": tsfn_latent if tsfn_latent else pl,
                    "expert_type": "tsfn",
                    "folder_path": FOLDER_TEMPLATE.format(
                        expert_type="tsfn", domain=domain, pl=pl,
                        expert=expert, sl=sl, ll=ll),
                })
            for expert in tsft:
                dim = TSFT_DIM.get(expert)
                if dim is None:
                    skipped.append(f"{expert}: unknown text expert, no hidden size")
                    continue
                rows.append({
                    "domain": domain, "pl": pl, "expert": expert, "freq": freq,
                    "latent_dim": dim, "expert_type": "tsft",
                    "folder_path": FOLDER_TEMPLATE.format(
                        expert_type="tsft", domain=domain, pl=pl,
                        expert=expert, sl=sl, ll=ll),
                })
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*",
                    default=["Economy", "Traffic", "SocialGood", "Security"],
                    help="folder names as they appear under MM-TSFlib/data/")
    ap.add_argument("--horizons", nargs="*", type=int, default=None,
                    help="default: every published horizon for each domain's frequency")
    ap.add_argument("--tsfn", nargs="*", default=TSFN_EXPERTS)
    ap.add_argument("--tsft", nargs="*", default=["GPT2", "BERT", "LLAMA2"])
    ap.add_argument("--tsfn-latent", type=int, default=None,
                    help="override the numeric latent width. Default: pred_len, "
                         "which is what the mm-mogu experts actually produce.")
    ap.add_argument("--all-monthly", action="store_true",
                    help="every monthly domain, overriding --domains")
    ap.add_argument("--out", default="all_experts_config.csv")
    args = ap.parse_args()

    domains = FREQ_GROUPS["monthly"][0] if args.all_monthly else args.domains
    rows, skipped = build(domains, args.horizons, args.tsfn, args.tsft, args.tsfn_latent)

    if not rows:
        sys.exit("nothing to write -- check --domains against MM-TSFlib/data/ "
                 "(note the repo's own spelling: 'Algriculture')")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_tsfn = sum(r["expert_type"] == "tsfn" for r in rows)
    n_tsft = len(rows) - n_tsfn
    print(f"wrote {args.out}")
    print(f"  {len(rows)} rows  ({n_tsfn} numeric, {n_tsft} text)")
    print(f"  domains:  {', '.join(domains)}")
    print(f"  horizons: {args.horizons or 'published defaults per frequency'}")
    if skipped:
        print("\nnotes:")
        for s in skipped[:10]:
            print(f"  ! {s}")

    print()
    print("=" * 70)
    print("READ THIS BEFORE USING IT FOR A GMM-TS 'direct' BASELINE")
    print("=" * 70)
    print("Text latent_dim values are real model hidden sizes (GPT2/BERT 768,")
    print("LLAMA2 4096) and are correct.")
    print()
    print("Numeric latent_dim is set to pred_len, which is what the mm-mogu experts")
    print("ACTUALLY produce: none of them returns (pred, latent), so the exp falls back")
    print("to flattening the prediction, and features='S' makes that pred_len wide.")
    print("For numeric experts, expert_input_type=latent and =prediction are therefore")
    print("numerically identical on this branch.")
    print()
    print("WHICH EXPERTS CAN DO agg_type=inv_var:")
    print("  PatchTST, iTransformer   -- have an UncHead, return (pred, sigma^2)")
    print("  DLinear, FiLM, Informer,")
    print("  Reformer                 -- NOT modified on mm-mogu; they return a plain")
    print("                              tensor, so prob_expert=1 crashes with")
    print("                              \"NoneType object is not subscriptable\".")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

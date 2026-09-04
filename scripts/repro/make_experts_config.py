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

Two consequences worth knowing:

  * With --expert_input_type prediction, Gating.py takes expert_dim = pred_len and
    NEVER reads latent_dim. The value in this file becomes irrelevant.
  * agg_type=inv_var is a separate, latent-free path (Gating.py line 175), so the
    expert projections are not used at all -- latent_dim is irrelevant there too,
    whatever expert_input_type says.

So: for the MM-MoGU (inv_var) runs this file just has to EXIST and have a row per
expert. For the GMM-TS (direct) baseline, run with expert_input_type=prediction
and the same holds. Only direct + latent needs a truthful latent_dim, and that is
the one combination this script cannot give you without the prerequisite stage.
It writes a documented placeholder and warns, rather than inventing a number and
staying quiet about it.

The text-expert dimensions below are architectural facts, not guesses:
GPT2/GPT2M/GPT2L/GPT2XL and BERT-base hidden sizes, and LLaMA-2-7B's 4096.
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

# Placeholder for numeric experts: d_model, which is the natural latent width for
# the TSLib-derived backbones. NOT verified against saved latents -- see the
# module docstring. Only consulted when expert_input_type == "latent".
TSFN_DEFAULT_LATENT = 512

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
                rows.append({
                    "domain": domain, "pl": pl, "expert": expert, "freq": freq,
                    "latent_dim": tsfn_latent, "expert_type": "tsfn",
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
    ap.add_argument("--tsfn-latent", type=int, default=TSFN_DEFAULT_LATENT,
                    help=f"placeholder numeric latent width (default {TSFN_DEFAULT_LATENT})")
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
    print(f"Text latent_dim values are real model hidden sizes and are correct.")
    print(f"Numeric latent_dim is a PLACEHOLDER ({args.tsfn_latent}), not measured from")
    print("saved latents. It is read only when --expert_input_type latent.")
    print()
    print("Safe combinations with this file:")
    print("  agg_type=inv_var   ANY expert_input_type   -- latent-free path, unused")
    print("  agg_type=direct    --expert_input_type prediction  -- uses pred_len instead")
    print()
    print("NOT safe: agg_type=direct with --expert_input_type latent. That sizes a")
    print("Linear from this placeholder and will either shape-mismatch or silently")
    print("train the wrong projection width. To do that properly, run the")
    print("save_tsfns_/save_tsfts_latents_predictions_*.sh prerequisite stage and")
    print("then scripts/dataset_prep_scripts/prepare_all_expert_config.py.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

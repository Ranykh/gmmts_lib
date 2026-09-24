"""Write all_experts_config.csv for the ONLINE gating path (run_online_gating.py).

Why this script exists
----------------------
The README builds all_experts_config.csv with
scripts/dataset_prep_scripts/prepare_all_expert_config.py, which reads latents saved by
`run.py --save_gating_dataset 1`. Against the pinned public MM-TSFlib
(e789ce78c9bafd8e3ba0d8850f9ad2becbe83548) that step cannot run: every MM-TSFlib model
returns a single tensor, while gmm_ts/exp/exp_long_term_forecasting.py unpacks
`outputs, _ = self.model(...)`, which raises
    ValueError: too many values to unpack (expected 2)
on the first training batch.

The online path only reads ONE number per (expert, domain, pl) from the CSV:
`latent_dim`, the input width of the gate's per-expert projection. With public
MM-TSFlib that width is fully determined by the online loop itself:

  * numeric (tsfn) experts: the model returns only its forecast, so the online loop
    uses the forecast as the latent -> shape (B, pred_len * c_out), and
    run_online_gating.py forces c_out = 1  =>  latent_dim = pred_len
  * text (tsft) expert: the pooled hidden layer of the gmmts text MLP, whose width is
    int(llm_dim / 8) (see MLP sizes in exp_online_gating_long_term_forecasting.py)

Caveat to report with any reproduced number: the GMM-TS authors' CSV came from MM-TSFlib
models that returned real backbone latents. With the public models the learned gate sees
each numeric expert's forecast in place of its latent. That is a property of the public
code, not of this script.

The output has exactly the schema prepare_all_expert_config.py writes, so
run_online_gating.py reads it unchanged.

Usage:
    python scripts/repro/make_online_experts_config.py --out_file all_experts_config.csv
"""
import argparse

import pandas as pd

# Forecasting setup from the GMM-TS paper / examples/*.sh (domain folder names as shipped
# by MM-TSFlib, including the upstream spelling "Algriculture").
SETUPS = {
    "daily": {"domains": ["Environment"], "pl": [48, 96, 192, 336]},
    "weekly": {"domains": ["Energy", "Public_Health"], "pl": [12, 24, 36, 48]},
    "monthly": {"domains": ["Algriculture", "Climate", "Economy", "Security",
                            "SocialGood", "Traffic"], "pl": [6, 8, 10, 12]},
}

TSFN_EXPERTS = ["Informer", "Reformer", "DLinear", "PatchTST", "FiLM", "iTransformer"]

# llm_dim exactly as run_online_gating.py assigns it. ClosedLLM is encoded with BERT and
# run_online_gating.py renames it to "BERT" before the CSV lookup, so it needs no row.
LLM_DIM = {"BERT": 768, "GPT2": 768, "GPT2M": 1024, "GPT2L": 1280, "GPT2XL": 1600,
           "LLAMA2": 4096, "LLAMA3": 4096}


def build_rows():
    rows = []
    for freq, setup in SETUPS.items():
        for domain in setup["domains"]:
            for pl in setup["pl"]:
                for expert in TSFN_EXPERTS:
                    rows.append({"domain": domain, "pl": pl, "expert": expert, "freq": freq,
                                 "latent_dim": pl, "expert_type": "tsfn",
                                 "folder_path": "online-no-saved-latents"})
                for llm, llm_dim in LLM_DIM.items():
                    rows.append({"domain": domain, "pl": pl, "expert": llm, "freq": freq,
                                 "latent_dim": int(llm_dim / 8), "expert_type": "tsft",
                                 "folder_path": "online-no-saved-latents"})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_file", type=str, default="all_experts_config.csv")
    args = parser.parse_args()

    df = pd.DataFrame(build_rows())
    # the online exp does int(expert_config['latent_dim']) on the filtered rows, which
    # needs exactly one row per (expert, domain, pl)
    assert not df.duplicated(["expert", "domain", "pl"]).any()
    df.to_csv(args.out_file, index=False)
    print("wrote {} rows to {}".format(len(df), args.out_file))

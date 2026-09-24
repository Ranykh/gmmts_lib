"""Collect online-gating runs produced by scripts/repro/run_online_sweep.sh.

Reads RUNS_DIR/<tag>/seed<seed>/<domain>/test_online_gating_results/<setting>/metrics.npy
(metric order is [mae, mse, rmse, mape, mspe] -- MSE is index 1), writes one long CSV row
per run, and prints the domain-level MSE averaged over horizons (and seeds) next to the
GMM-TS paper's Table 1.

Usage:
    python scripts/repro/collect_online_results.py --runs_dir runs --out results_online.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# GMM-TS paper, Table 1 (PatchTST + GPT2, MSE averaged over the four horizons)
TABLE1 = pd.DataFrame(
    [("Algriculture", 0.10, 0.11, 0.09), ("Climate", 1.32, 1.15, 1.02),
     ("Economy", 0.02, 0.04, 0.02), ("Energy", 0.28, 0.29, 0.27),
     ("Environment", 0.52, 0.47, 0.41), ("Public_Health", 1.61, 1.46, 1.17),
     ("Security", 116.43, 112.90, 110.30), ("SocialGood", 1.14, 0.99, 0.95),
     ("Traffic", 0.21, 0.20, 0.19)],
    columns=["domain", "paper_unimodal", "paper_timemmd", "paper_gmmts"])

SETTING_RE = re.compile(
    r"^(?P<domain>.+)_long_term_forecast_agg(?P<agg>[^_]+)_eip(?P<eip>[^_]+)_pl(?P<pl>\d+)_"
    r".*_tsfn-experts(?P<tsfn>.+)_tsft-experts(?P<llm>.+)$")


def collect(runs_dir):
    rows = []
    pattern = os.path.join(runs_dir, "*", "seed*", "*", "test_online_gating_results", "*",
                           "metrics.npy")
    for path in sorted(glob.glob(pattern)):
        result_dir = os.path.dirname(path)
        setting = os.path.basename(result_dir)
        parts = result_dir.split(os.sep)
        tag, seed = parts[-5], parts[-4].replace("seed", "")
        m = SETTING_RE.match(setting)
        if m is None:
            print("skipping unrecognised setting:", setting)
            continue
        mae, mse, rmse, mape, mspe = np.load(path)
        row = dict(tag=tag, seed=int(seed), domain=m["domain"], pl=int(m["pl"]),
                   agg=m["agg"], tsfn=m["tsfn"], llm=m["llm"], mse=float(mse), mae=float(mae))
        # MoGU runs also store per-expert forecasts and gate weights
        names_file = os.path.join(result_dir, "expert_names.txt")
        if os.path.exists(names_file):
            names = open(names_file).read().split()
            true = np.load(os.path.join(result_dir, "true_scaled.npy"))           # (N, H)
            expert_pred = np.load(os.path.join(result_dir, "expert_pred.npy"))    # (N, E, H)
            weights = np.load(os.path.join(result_dir, "gate_weights.npy"))       # (N, E, H)
            for e, name in enumerate(names):
                row["mse_" + name] = float(np.mean((expert_pred[:, e] - true) ** 2))
                row["w_" + name] = float(weights[:, e].mean())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--out", type=str, default="results_online.csv")
    args = parser.parse_args()

    df = collect(args.runs_dir)
    if df.empty:
        raise SystemExit("no metrics.npy found under {}".format(args.runs_dir))
    df = df.sort_values(["tag", "domain", "pl", "seed"])
    df.to_csv(args.out, index=False)
    print("wrote {} runs to {}\n".format(len(df), args.out))

    # runs per (tag, domain) -- a full domain is 4 horizons x #seeds
    counts = df.groupby(["tag", "domain"]).size().rename("runs")
    per_domain = (df.groupby(["tag", "domain"])["mse"].mean().rename("mse_mean")
                  .to_frame().join(counts).reset_index())
    table = per_domain.merge(TABLE1, on="domain", how="left")
    table["vs_paper_gmmts_%"] = 100 * (table["mse_mean"] / table["paper_gmmts"] - 1)
    with pd.option_context("display.width", 160, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))

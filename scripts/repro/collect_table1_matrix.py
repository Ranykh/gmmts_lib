"""Build the Table-1-shaped comparison from a run_table1_matrix.py runs directory.

Per domain it reports test MSE averaged over the domain's horizons (then mean +- std over
seeds) for every arm, next to the GMM-TS paper's Table 1:

  Unimodal, Time-MMD (pw picked per domain on validation MSE), GMM-TS (learned gate),
  MM-MoGU (MoGU's own checkpoint rule: best validation NLL), MM-MoGU selected like GMM-TS
  (best validation MSE of the gated forecast), and MM-MoGU with detached weights if it ran.

Diagnostics, because MSE alone can hide a gate that does nothing:
  * Time-MMD as upstream's test() would score it (the fusion applied twice)
  * MoGU vs GMM-TS: paired over (horizon, seed) -- wins and a paired t-test
  * gate weight std across test samples (near 0 = a constant blend, not routing), and
    whether the gated forecast beats its own best single expert

Never averages MSE across domains: scales differ by >1000x.

Usage:
    python scripts/repro/collect_table1_matrix.py --runs_dir runs/table1_matrix
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# GMM-TS paper, Table 1 (PatchTST + GPT2; MSE averaged over the four horizons)
PAPER = pd.DataFrame(
    [("Algriculture", "Agriculture", 0.10, 0.23, 0.11, 0.09),
     ("Climate", "Climate", 1.32, 1.27, 1.15, 1.02),
     ("Economy", "Economy", 0.02, 0.02, 0.04, 0.02),
     ("Energy", "Energy", 0.28, 0.28, 0.29, 0.27),
     ("Environment", "Environment", 0.52, 0.59, 0.47, 0.41),
     ("Public_Health", "Public Health", 1.61, 1.96, 1.46, 1.17),
     ("Security", "Security", 116.43, 74.91, 112.90, 110.30),
     ("SocialGood", "Social Good", 1.14, 1.13, 0.99, 0.95),
     ("Traffic", "Traffic", 0.21, 0.22, 0.20, 0.19)],
    columns=["domain", "Domain", "paper_unimodal", "paper_gpt4mts", "paper_timemmd", "paper_gmmts"])


MOGU_VARIANTS = ("mogu", "mogu_mse", "mogu_detached")
LABELS = {"mogu": "MM-MoGU", "mogu_mse": "MoGU(MSE)", "mogu_detached": "MoGU(detach)"}


def _gate_stats(folder, weights_file="gate_weights.npy"):
    """Gate weight spread and gated-vs-best-expert, from the files test() saves."""
    wf = folder / weights_file
    if not (wf.exists() and (folder / "expert_pred.npy").exists()):
        return {}
    w = np.load(wf)                                   # (N, E, H)
    mu = np.load(folder / "expert_pred.npy")          # (N, E, H)
    y = np.load(folder / "true_scaled.npy")           # (N, H)
    names = (folder / "expert_names.txt").read_text().split()
    own = [float(np.mean((mu[:, e] - y) ** 2)) for e in range(mu.shape[1])]
    out = {"gate_w_std": float(np.mean([w[:, e].std() for e in range(w.shape[1])])),
           "best_expert": names[int(np.argmin(own))], "best_expert_mse": min(own)}
    for e, n in enumerate(names):
        out["w_" + n] = float(w[:, e].mean())
        out["mse_" + n] = own[e]
    return out


def load_runs(runs_dir):
    rows = []
    for rec_file in sorted(Path(runs_dir).glob("*/seed*/*/pl*/run.json")):
        rec = json.loads(rec_file.read_text())
        if rec.get("status") != "ok":
            continue
        job_dir = rec_file.parent
        folder = (job_dir / rec["metrics_file"]).parent
        mae, mse = np.load(folder / "metrics.npy")[:2]
        row = {k: rec[k] for k in ("arm", "label", "prompt_weight", "domain", "pl", "seed",
                                   "tsfn", "llm", "epochs", "best_vali_loss", "seconds",
                                   "code_commit", "mmtsflib_commit")}
        row.update(mse=float(mse), mae=float(mae))
        if (folder / "metrics_upstream_test.npy").exists():
            row["mse_upstream_test"] = float(np.load(folder / "metrics_upstream_test.npy")[1])
        if (folder / "metrics_select_mse.npy").exists():
            row["mse_select_mse"] = float(np.load(folder / "metrics_select_mse.npy")[1])
            row["gate_w_std_select_mse"] = _gate_stats(
                folder, "gate_weights_select_mse.npy").get("gate_w_std")
        row.update(_gate_stats(folder))
        if "best_expert_mse" in row:
            row["gate_beats_best_expert"] = bool(row["mse"] < row["best_expert_mse"])
        rows.append(row)
    return pd.DataFrame(rows)


def pick_timemmd_pw(runs):
    """Time-MMD tunes its fusion weight per domain: pick the pw with the lowest mean
    validation MSE over that domain's horizons and seeds."""
    t = runs[runs.arm == "timemmd"]
    if t.empty:
        return {}
    v = t.groupby(["domain", "prompt_weight"])["best_vali_loss"].mean().reset_index()
    return {d: float(g.loc[g.best_vali_loss.idxmin(), "prompt_weight"]) for d, g in v.groupby("domain")}


def domain_mean(df, col):
    """Horizon-mean per seed, then mean/std over seeds. Returns (mean, std, n_seeds, n_runs)."""
    if df.empty or col not in df or df[col].isna().all():
        return np.nan, np.nan, 0, 0
    per_seed = df.groupby("seed")[col].mean()
    return (float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
            int(len(per_seed)), int(df[col].notna().sum()))


def paired(a, b, col="mse"):
    """Paired comparison of arm a vs arm b over matching (pl, seed): wins of a, paired t-test."""
    m = a[["pl", "seed", col]].merge(b[["pl", "seed", col]], on=["pl", "seed"],
                                     suffixes=("_a", "_b")).dropna()
    if m.empty:
        return {}
    xa, xb = m[col + "_a"], m[col + "_b"]
    out = {"pairs": int(len(m)), "wins": int((xa < xb).sum())}
    if len(m) > 1:
        try:
            from scipy.stats import ttest_rel
            out["p_paired"] = float(ttest_rel(xa, xb).pvalue)
        except ImportError:
            pass
    return out


def summarize(runs):
    chosen = pick_timemmd_pw(runs)
    arms = {
        "unimodal": (runs[runs.arm == "unimodal"], "mse"),
        "timemmd": (runs[(runs.arm == "timemmd") &
                         (runs.prompt_weight == runs.domain.map(chosen))], "mse"),
        "timemmd_upstream_test": (runs[(runs.arm == "timemmd") &
                                       (runs.prompt_weight == runs.domain.map(chosen))],
                                  "mse_upstream_test"),
        "gmmts": (runs[runs.arm == "gmmts"], "mse"),
        "mogu": (runs[runs.arm == "mogu"], "mse"),
        "mogu_select_mse": (runs[runs.arm == "mogu"], "mse_select_mse"),
        "mogu_detached": (runs[runs.arm == "mogu_detached"], "mse"),
        "mogu_mse": (runs[runs.arm == "mogu_mse"], "mse"),
    }
    rows = []
    for d in PAPER.domain:
        row = {"domain": d}
        for name, (df, col) in arms.items():
            mean, std, n_seeds, n_runs = domain_mean(df[df.domain == d], col)
            row[name] = mean
            row[name + "_std"] = std
            row[name + "_runs"] = n_runs
        row["timemmd_pw"] = chosen.get(d, np.nan)
        g = runs[(runs.arm == "gmmts") & (runs.domain == d)]
        for v in MOGU_VARIANTS:
            pr = paired(runs[(runs.arm == v) & (runs.domain == d)], g)
            row[v + "_vs_gmmts_%"] = 100 * (row[v] / row["gmmts"] - 1) if row["gmmts"] else np.nan
            row[v + "_wins"] = "{}/{}".format(pr["wins"], pr["pairs"]) if pr else ""
            row[v + "_vs_gmmts_p"] = pr.get("p_paired", np.nan)
        for arm in ("gmmts",) + MOGU_VARIANTS:
            sub = runs[(runs.arm == arm) & (runs.domain == d)]
            row["gate_w_std_" + arm] = sub["gate_w_std"].mean() if "gate_w_std" in sub else np.nan
            row["beats_best_expert_" + arm] = (
                "{}/{}".format(int(sub.gate_beats_best_expert.sum()), int(sub.gate_beats_best_expert.notna().sum()))
                if "gate_beats_best_expert" in sub and sub.gate_beats_best_expert.notna().any() else "")
        row["seeds"] = int(runs[runs.domain == d].seed.nunique())
        rows.append(row)
    s = PAPER.merge(pd.DataFrame(rows), on="domain")
    s["gmmts_vs_paper_%"] = 100 * (s["gmmts"] / s["paper_gmmts"] - 1)
    return s, chosen


def _fmt(mean, std):
    if pd.isna(mean):
        return "-"
    digits = 2 if abs(mean) >= 10 else 4
    return ("{:.%df}" % digits).format(mean) + ("" if pd.isna(std) else ("±{:.%df}" % digits).format(std))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs_dir", default="runs/table1_matrix")
    p.add_argument("--out", default=None, help="output prefix (default: <runs_dir>/table1_matrix)")
    a = p.parse_args(argv)
    runs = load_runs(a.runs_dir)
    if runs.empty:
        print("no finished runs under", a.runs_dir)
        return 1
    prefix = a.out or str(Path(a.runs_dir) / "table1_matrix")
    summary, chosen = summarize(runs)
    runs.sort_values(["domain", "arm", "prompt_weight", "pl", "seed"]).to_csv(prefix + "_runs.csv", index=False)
    summary.to_csv(prefix + "_summary.csv", index=False)

    view = pd.DataFrame({"Domain": summary.Domain,
                         "paper Uni": summary.paper_unimodal, "paper TimeMMD": summary.paper_timemmd,
                         "paper GMM-TS": summary.paper_gmmts})
    for name, label in [("unimodal", "Unimodal"), ("timemmd", "TimeMMD"), ("gmmts", "GMM-TS"),
                        ("mogu", "MM-MoGU"), ("mogu_select_mse", "MoGU(selMSE)"),
                        ("mogu_mse", "MoGU(MSE)"), ("mogu_detached", "MoGU(detach)")]:
        if summary[name + "_runs"].sum() > 0:
            view[label] = [_fmt(m, s) for m, s in zip(summary[name], summary[name + "_std"])]
    view["pw"] = summary.timemmd_pw
    diag = pd.DataFrame({"Domain": summary.Domain,
                         "TimeMMD upstream test()": [_fmt(m, s) for m, s in zip(summary.timemmd_upstream_test, summary.timemmd_upstream_test_std)],
                         "w std GMM-TS": summary.gate_w_std_gmmts.round(3),
                         "GMM-TS beats best expert": summary.beats_best_expert_gmmts})
    for v in MOGU_VARIANTS:
        if summary[v + "_runs"].sum() == 0:
            continue
        lab = LABELS[v]
        diag[lab + " vs GMM-TS %"] = summary[v + "_vs_gmmts_%"].round(1)
        diag[lab + " wins"] = summary[v + "_wins"]
        diag[lab + " p"] = summary[v + "_vs_gmmts_p"].round(3)
        diag["w std " + lab] = summary["gate_w_std_" + v].round(3)
        diag[lab + " beats best expert"] = summary["beats_best_expert_" + v]
    diag["seeds"] = summary.seeds
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nTest MSE per domain, mean over horizons (± std over seeds). Experts: {} + {}.".format(
            runs.tsfn.iloc[0], runs.llm.iloc[0]))
        print(view.to_string(index=False))
        print("\nDiagnostics")
        print(diag.to_string(index=False))
    print("\nwrote {}_summary.csv and {}_runs.csv ({} runs)".format(prefix, prefix, len(runs)))
    try:
        with pd.ExcelWriter(prefix + ".xlsx") as xw:
            view.to_excel(xw, sheet_name="Table 1 view", index=False)
            diag.to_excel(xw, sheet_name="Diagnostics", index=False)
            summary.to_excel(xw, sheet_name="Summary (numbers)", index=False)
            runs.to_excel(xw, sheet_name="All runs", index=False)
        print("wrote {}.xlsx".format(prefix))
    except ImportError:
        print("(pip install openpyxl to also get an .xlsx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

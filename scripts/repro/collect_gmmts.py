#!/usr/bin/env python3
"""Collect gmmts_lib online-gating runs into one tidy CSV.

    python scripts/repro/collect_gmmts.py                    # scan ./test_online_gating_results
    python scripts/repro/collect_gmmts.py --out gmmts_g4.csv
    python scripts/repro/collect_gmmts.py --check            # audit only, write nothing

What it reads
-------------
run_mmmogu_sweep.sh renames each results folder to

    {domain}_long_term_forecast_agg{agg}_eip{eip}_pl{N}_dm{N}_nh{N}_el{N}_df{N}_{itr}
    _tsfn-experts{TSF-N}_tsft-experts{TSF-T}__norm-{inv_var_norm}__pe{0|1}__seed{N}

The part before `__norm-` is what run_online_gating.py builds at line 210. The
three trailing fields are appended by the sweep script because the repo's own
setting string omits them -- see below.

metrics.npy is np.array([mae, mse, rmse, mape, mspe]) -- MAE FIRST, not MSE.

The overwrite trap, and why folders look odd
--------------------------------------------
run_online_gating.py derives `domain` from the last path segment of --root_path
(line 152) and puts neither seed nor prob_expert nor inv_var_norm into the
folder name. Worse, line 223 SKIPS any setting whose folder already exists. So
without the rename, `inv_var_norm=none` and `inv_var_norm=per_modality` collide,
and the second one is never run -- the log cheerfully says "has been trained and
tested, skip it".

This script therefore treats a folder WITHOUT the `__norm-...__pe...__seed...`
suffix as suspect: it was written before the rename, so its seed and
normalisation are unknown and it cannot be told apart from a sibling config.

Gate weights are not available
------------------------------
exp_online_gating_long_term_forecasting.py saves metrics.npy, pred.npy and
true.npy only. It never saves the gate weights, even though
GatingNet.forward(data, return_w=True) returns them for every aggregation type
including inv_var. So a gate that has collapsed to a constant blend is
indistinguishable here from one that is genuinely routing -- and a collapsed
gate with a good MSE is worse than a loss, because it looks publishable and is
wrong. This script says so out loud rather than leaving an empty column that
reads as benign.
"""
import argparse
import csv
import datetime as _dt
import os
import re
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("needs numpy: pip install numpy")


SETTING_RE = re.compile(
    r"^(?P<domain>.+?)_(?P<task>long_term_forecast)_"
    r"agg(?P<agg_type>[A-Za-z_]+)_"
    r"eip(?P<expert_input_type>[A-Za-z]+)_"
    r"pl(?P<pred_len>\d+)_"
    r"dm(?P<gating_d_model>\d+)_"
    r"nh(?P<n_heads>\d+)_"
    r"el(?P<e_layers>\d+)_"
    r"df(?P<d_ff>\d+)_"
    r"(?P<itr>\d+)_"
    r"tsfn-experts(?P<tsf_n>[A-Za-z0-9-]+)_"
    r"tsft-experts(?P<tsf_t>[A-Za-z0-9.]+)"
    r"(?:__norm-(?P<inv_var_norm>[A-Za-z_]+)"
    r"__pe(?P<prob_expert>\d)"
    r"__seed(?P<seed>\d+))?$"
)

# Folder domain -> the label the comparison workbook uses.
DOMAIN_TO_SHEET = {
    "Algriculture": "Agriculture",       # the repo's own spelling
    "Climate": "Climate",
    "Economy": "Economy",
    "Energy": "Energy",
    "Environment": "Environment",
    "Public_Health": "Public Health",
    "Security": "Security",
    "SocialGood": "Social Good",
    "Traffic": "Traffic",
}

# (agg_type, inv_var_norm) -> the label used everywhere downstream.
#
# inv_var MUST be split by inv_var_norm. Both variants write agg_type=inv_var,
# so collapsing them to one label would silently average two different methods
# into one cell.
#
#   none          raw 1/sigma^2                       -> G4_IV
#   per_modality  log-variance z-scored WITHIN each    -> G4n_IV_permod
#                 modality (Gating.py:136)
#
# G4n is NOT the workbook's G5 SNIV. G5 is w ~ (sigma^-2 / s_i) with
# s_i = E_val[sigma_i^-2], a per-EXPERT validation estimate. A per-modality
# z-score is a different operation -- and one that is provably invariant to
# per-expert variance rescaling, so it cannot substitute for calibration.
AGG_TO_METHOD = {
    "direct": "G3_ATTN_direct",
    "latent": "G3_ATTN_latent",
    "hierarchical": "G3_ATTN_hierarchical",
    "inv_var": "G4_IV",
}


def method_of(agg, inv_var_norm):
    if agg == "inv_var" and inv_var_norm == "per_modality":
        return "G4n_IV_permod"
    return AGG_TO_METHOD.get(agg, agg)

FIELDS = ["repo", "git_sha", "git_dirty", "mm_tsflib_sha",
          "method", "domain", "domain_sheet", "tsf_n", "tsf_t",
          "agg_type", "prob_expert", "inv_var_norm", "expert_input_type",
          "pred_len", "seed",
          "mse", "mae", "rmse", "mape", "mspe",
          "gate_w_std", "name_complete", "setting", "run_dir", "mtime"]


def git_info(root):
    def _run(a):
        return subprocess.run(a, cwd=root, capture_output=True, text=True, timeout=10)
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        if sha.returncode != 0:
            return "NO-GIT", ""
        st = _run(["git", "status", "--porcelain"])
        return sha.stdout.strip(), ("yes" if st.stdout.strip() else "no")
    except (OSError, subprocess.SubprocessError):
        return "NO-GIT", ""


def read_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.npy")
    if not os.path.exists(path):
        return None
    try:
        m = np.load(path)
    except (OSError, ValueError):
        return None
    if m.size < 2:
        return None
    names = ["mae", "mse", "rmse", "mape", "mspe"]
    return {n: (float(m[i]) if i < m.size else "") for i, n in enumerate(names)}


def read_gate(run_dir):
    """Present only if someone has added the save. Absent in the repo as committed."""
    for fn in ("gate_weights.npy", "weights.npy"):
        path = os.path.join(run_dir, fn)
        if os.path.exists(path):
            try:
                w = np.load(path)
                if w.ndim >= 2:
                    return round(float(w.std(axis=0).mean()), 6)
            except (OSError, ValueError):
                return ""
    return ""


def scan(results_root, repo_root, mm_sha):
    sha, dirty = git_info(repo_root)
    rows, unparsed, incomplete = [], [], []

    if not os.path.isdir(results_root):
        return rows, unparsed, incomplete, sha

    for name in sorted(os.listdir(results_root)):
        run_dir = os.path.join(results_root, name)
        if not os.path.isdir(run_dir):
            continue
        m = SETTING_RE.match(name)
        if not m:
            unparsed.append(name)
            continue
        d = m.groupdict()
        metrics = read_metrics(run_dir)
        if metrics is None:
            unparsed.append(f"{name}  (no readable metrics.npy)")
            continue

        complete = d["seed"] is not None
        if not complete:
            incomplete.append(name)

        agg = d["agg_type"]
        row = {
            "repo": "gmmts_lib",
            "git_sha": sha, "git_dirty": dirty, "mm_tsflib_sha": mm_sha,
            "method": method_of(agg, d["inv_var_norm"] or ""),
            "domain": d["domain"],
            "domain_sheet": DOMAIN_TO_SHEET.get(d["domain"], d["domain"]),
            "tsf_n": d["tsf_n"], "tsf_t": d["tsf_t"],
            "agg_type": agg,
            "prob_expert": int(d["prob_expert"]) if d["prob_expert"] else "",
            "inv_var_norm": d["inv_var_norm"] or "",
            "expert_input_type": d["expert_input_type"],
            "pred_len": int(d["pred_len"]),
            "seed": int(d["seed"]) if d["seed"] else "",
            "gate_w_std": read_gate(run_dir),
            "name_complete": "yes" if complete else "NO",
            "setting": name, "run_dir": run_dir,
            "mtime": _dt.datetime.fromtimestamp(
                os.path.getmtime(run_dir)).isoformat(timespec="seconds"),
        }
        row.update(metrics)
        # An inv_var run with prob_expert=0 is impossible -- run_online_gating.py
        # guards it -- so seeing one means the folder name lies.
        if agg == "inv_var" and row["prob_expert"] == 0:
            incomplete.append(f"{name}  (inv_var with prob_expert=0 -- name is wrong)")
        rows.append(row)

    rows.sort(key=lambda r: (r["domain_sheet"], r["pred_len"], r["tsf_n"],
                             r["tsf_t"], r["method"], str(r["seed"])))
    return rows, unparsed, incomplete, sha


def audit(rows, unparsed, incomplete, results_root):
    print("=" * 70)
    print("AUDIT")
    print("=" * 70)
    print(f"results dir:  {results_root}")
    print(f"runs parsed:  {len(rows)}")
    fatal = False

    if unparsed:
        print(f"\nunparsed folders: {len(unparsed)}")
        for u in unparsed[:8]:
            print(f"    {u}")

    if incomplete:
        fatal = True
        print(f"\n!! {len(incomplete)} folder(s) lack the __norm-..__pe..__seed.. suffix,")
        print("!! or contradict it. run_online_gating.py does not put seed, prob_expert")
        print("!! or inv_var_norm in the setting string, and it SKIPS a setting whose")
        print("!! folder already exists -- so these cannot be told apart from a sibling")
        print("!! config, and a colliding run may never have executed at all.")
        for u in incomplete[:8]:
            print(f"!!   {u}")
        print("!! Re-run them through scripts/repro/run_mmmogu_sweep.sh, which renames.")

    # Collision check on the full experiment key.
    seen = {}
    for r in rows:
        k = (r["domain"], r["tsf_n"], r["tsf_t"], r["pred_len"],
             r["agg_type"], r["inv_var_norm"], r["prob_expert"], r["seed"])
        seen.setdefault(k, []).append(r)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        fatal = True
        print(f"\n!! {len(dupes)} experiment key(s) appear more than once:")
        for k, v in list(dupes.items())[:5]:
            print(f"!!   {k} x{len(v)}")

    if rows and not any(r["gate_w_std"] != "" for r in rows):
        print("\n!  NO GATE WEIGHTS in any run. The online exp saves metrics/pred/true")
        print("!  only, so a gate that collapsed to a constant blend looks exactly like")
        print("!  one that is routing. The MSE cannot tell you which you have.")
        print("!  GatingNet.forward(data, return_w=True) already returns the weights --")
        print("!  saving them is a small flag-guarded change to")
        print("!  gmm_ts/exp/exp_online_gating_long_term_forecasting.py.")

    if rows and rows[0]["git_dirty"] == "yes":
        print("\n!! git_dirty=yes -- results are not tied to the recorded SHA.")

    # Pairing check: the G4-vs-G3 claim needs both halves of each cell.
    g3 = {(r["domain"], r["tsf_n"], r["tsf_t"], r["pred_len"], r["seed"])
          for r in rows if r["agg_type"] == "direct"}
    g4 = {(r["domain"], r["tsf_n"], r["tsf_t"], r["pred_len"], r["seed"])
          for r in rows if r["method"] == "G4_IV"}
    both, only3, only4 = g3 & g4, g3 - g4, g4 - g3
    print(f"\npaired cells (G3 and G4 both present): {len(both)}")
    if only3 or only4:
        print(f"  G3 only: {len(only3)}   G4 only: {len(only4)}")
        print("  An unpaired cell cannot support the apples-to-apples comparison.")
    if both:
        wins = 0
        m3 = {(r["domain"], r["tsf_n"], r["tsf_t"], r["pred_len"], r["seed"]): r["mse"]
              for r in rows if r["agg_type"] == "direct"}
        m4 = {(r["domain"], r["tsf_n"], r["tsf_t"], r["pred_len"], r["seed"]): r["mse"]
              for r in rows if r["method"] == "G4_IV"}
        for k in both:
            if m4[k] < m3[k]:
                wins += 1
        print(f"  MM-MoGU (G4) beats our GMM-TS (G3) in {wins}/{len(both)} "
              f"({wins / len(both):.0%}) of paired cells.")
        print("  Report this honestly. Winning everywhere is as suspicious as")
        print("  winning nowhere -- MoGU wins 18 of 32 in its own paper.")
    print("=" * 70)
    return fatal


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    ap.add_argument("--results",
                    default=os.path.join(repo_root, "test_online_gating_results"))
    ap.add_argument("--repo-root", default=repo_root)
    ap.add_argument("--mm-tsflib",
                    default=os.environ.get("MM_TSFLIB_PATH",
                                           os.path.expanduser("~/msc/MM-TSFlib")))
    ap.add_argument("--out", default="gmmts_results.csv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="write the CSV even if the audit found collisions")
    args = ap.parse_args()

    mm_sha, _ = git_info(args.mm_tsflib)
    rows, unparsed, incomplete, sha = scan(args.results, args.repo_root, mm_sha)
    if not rows:
        sys.exit(f"no runs found under {args.results}")

    fatal = audit(rows, unparsed, incomplete, args.results)

    if args.check:
        print("\n--check: nothing written.")
        return 1 if fatal else 0
    if fatal and not args.force:
        print("\nREFUSING to write a CSV the audit flagged. Re-run the affected")
        print("configs through the sweep script, or pass --force deliberately.")
        return 1

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

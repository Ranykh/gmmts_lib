"""Run the GMM-TS Table 1 comparison matrix in one command: Unimodal, Time-MMD, GMM-TS and
MM-MoGU, all from this repository, on the same data, splits, horizons and seeds.

  arm            entry point                      method
  -------------  -------------------------------  -------------------------------------------
  unimodal       run.py, --prompt_weight 0        numeric expert alone (Table 1 "Unimodal")
  timemmd        run.py, --prompt_weight pw       Time-MMD fusion (1-pw)*numeric + pw*text, one
                                                  run per pw in --timemmd_pw; the collector picks
                                                  pw per domain on validation MSE
  gmmts          run_online_gating.py, direct     GMM-TS learned gate (Table 1 "GMM-TS")
  mogu           run_online_gating.py, mogu       MM-MoGU inverse-variance gate
  mogu_mse       mogu + --mogu_loss mse: forecasts trained exactly as in GMM-TS, variance
                 heads by NLL only -- the gate is the only difference from gmmts
  mogu_detached  (opt-in) mogu + --mogu_detach_weights 1

Experts default to Table 1's pair: --tsfn PatchTST, --llm GPT2.

One job = one (arm, domain, horizon, seed), run in its own directory
RUNS_DIR/<arm>/seed<seed>/<domain>/pl<pl>/ with log.txt and run.json. Jobs are spread over
--gpus (--jobs_per_gpu processes per card), longest first. Re-running the same command skips
finished jobs, so an interrupted session resumes where it stopped. At the end
collect_table1_matrix.py writes the Table-1-shaped summary (CSV, and XLSX if openpyxl is
installed).

Usage (repo root, venv active, MM_TSFLIB_PATH set, inside tmux):
    python scripts/repro/run_table1_matrix.py --gpus 2,5 --seeds 2021 2022 2023
"""
import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

# Table 1 domains in the paper's order; folder names as shipped by MM-TSFlib
# (csv, frequency, seq_len, label_len, horizons) -- the paper's forecasting setup
DOMAINS = {
    "Algriculture": ("US_RetailBroilerComposite_Month.csv", "monthly", 8, 4, [6, 8, 10, 12]),
    "Climate": ("US_precipitation_month.csv", "monthly", 8, 4, [6, 8, 10, 12]),
    "Economy": ("US_TradeBalance_Month.csv", "monthly", 8, 4, [6, 8, 10, 12]),
    "Energy": ("US_GasolinePrice_Week.csv", "weekly", 36, 18, [12, 24, 36, 48]),
    "Environment": ("NewYork_AQI_Day.csv", "daily", 96, 48, [48, 96, 192, 336]),
    "Public_Health": ("US_FLURATIO_Week.csv", "weekly", 36, 18, [12, 24, 36, 48]),
    "Security": ("US_FEMAGrant_Month.csv", "monthly", 8, 4, [6, 8, 10, 12]),
    "SocialGood": ("Unadj_UnemploymentRate_ALL_processed.csv", "monthly", 8, 4, [6, 8, 10, 12]),
    "Traffic": ("US_VMT_Month.csv", "monthly", 8, 4, [6, 8, 10, 12]),
}
# relative run time (batches per run), only used to start the longest jobs first
COST = {"daily": 30.0, "weekly": 4.0, "monthly": 1.0}
ARMS = ("unimodal", "timemmd", "gmmts", "mogu", "mogu_detached", "mogu_mse")


class Job:
    def __init__(self, arm, pw, domain, pl, seed, runs_dir):
        self.arm, self.pw, self.domain, self.pl, self.seed = arm, pw, domain, pl, seed
        self.label = "timemmd_pw{}".format(pw) if arm == "timemmd" else arm
        self.dir = runs_dir / self.label / "seed{}".format(seed) / domain / "pl{}".format(pl)
        self.cost = COST[DOMAINS[domain][1]] * (1 + pl / 100.0)

    def done(self):
        rec = self.dir / "run.json"
        if not rec.exists():
            return False
        try:
            return json.loads(rec.read_text()).get("status") == "ok"
        except ValueError:
            return False


def git_rev(path):
    try:
        rev = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                         "--untracked-files=no"], text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_cmd(job, a, mm_path, config):
    csv, _, seq_len, label_len, _ = DOMAINS[job.domain]
    common = ["--task_name", "long_term_forecast", "--is_training", "1",
              "--root_path", os.path.join(mm_path, "data", job.domain), "--data_path", csv,
              "--model_id", "{}_{}_s{}_pl{}".format(job.domain, job.label, job.seed, job.pl),
              "--data", "custom", "--features", "M",
              "--seq_len", str(seq_len), "--label_len", str(label_len), "--pred_len", str(job.pl),
              "--des", "Exp", "--seed", str(job.seed),
              "--type_tag", "#F#", "--text_len", "4", "--pool_type", "avg",
              "--llm_model", a.llm, "--use_fullmodel", "0",
              "--train_epochs", str(a.epochs), "--save_name", "results.txt", "--gpu", "0"]
    if job.arm in ("unimodal", "timemmd"):
        cmd = [sys.executable, "-u", str(REPO / "run.py"), "--model", a.tsfn,
               "--prompt_weight", str(job.pw)]
    else:
        cmd = [sys.executable, "-u", str(REPO / "run_online_gating.py"), "--model", a.tsfn,
               "--agg_type", "direct" if job.arm == "gmmts" else "mogu",
               "--all_experts_config", str(config)]
        if job.arm == "mogu_detached":
            cmd += ["--mogu_detach_weights", "1"]
        if job.arm == "mogu_mse":
            cmd += ["--mogu_loss", "mse"]
    return cmd + common + shlex.split(a.extra)


def run_job(job, gpu, a, mm_path, config, provenance):
    if job.dir.exists():
        shutil.rmtree(job.dir)  # a failed or partial earlier attempt: start clean
    job.dir.mkdir(parents=True)
    cmd = build_cmd(job, a, mm_path, config)
    env = dict(os.environ, MM_TSFLIB_PATH=mm_path,
               CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else str(gpu))
    start = time.time()
    with open(job.dir / "log.txt", "w") as log:
        rc = subprocess.call(cmd, cwd=job.dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    text = (job.dir / "log.txt").read_text(errors="replace")
    vali = [float(v) for v in re.findall(r"Vali Loss: ([-+0-9.eE]+|nan)", text)]
    metrics = sorted(job.dir.glob("**/metrics.npy"))
    ok = rc == 0 and len(metrics) == 1
    record = dict(arm=job.arm, label=job.label, prompt_weight=job.pw, domain=job.domain,
                  pl=job.pl, seed=job.seed, tsfn=a.tsfn, llm=a.llm, epochs=a.epochs,
                  status="ok" if ok else "failed", returncode=rc, seconds=round(seconds, 1),
                  gpu=gpu, best_vali_loss=(min(vali) if vali else None),
                  metrics_file=(str(metrics[0].relative_to(job.dir)) if metrics else None),
                  cmd=cmd, finished=time.strftime("%Y-%m-%d %H:%M:%S"), **provenance)
    (job.dir / "run.json").write_text(json.dumps(record, indent=1))
    mse = float(np.load(metrics[0])[1]) if ok else None
    return ok, seconds, mse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpus", default="0", help="comma list of card ids, e.g. 2,5 ('cpu' = no GPU)")
    p.add_argument("--jobs_per_gpu", type=int, default=2,
                   help="processes per card; runs are small and tokenizer (CPU) bound")
    p.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    p.add_argument("--arms", nargs="+", default=["unimodal", "timemmd", "gmmts", "mogu", "mogu_mse"],
                   choices=ARMS)
    p.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument("--horizons", nargs="+", type=int, default=None,
                   help="debug only: restrict every domain to these horizons")
    p.add_argument("--timemmd_pw", nargs="+", type=float, default=[0.01, 0.1, 0.2],
                   help="Time-MMD fusion weights to run; picked per domain on validation")
    p.add_argument("--tsfn", default="PatchTST", help="numeric expert(s), comma list for gmmts/mogu")
    p.add_argument("--llm", default="GPT2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--runs_dir", default="runs/table1_matrix")
    p.add_argument("--extra", default="", help='extra args for every run, e.g. "--num_workers 0"')
    p.add_argument("--dry_run", action="store_true", help="print the plan and exit")
    a = p.parse_args(argv)

    mm_path = os.environ.get("MM_TSFLIB_PATH")
    if not mm_path:
        sys.exit("export MM_TSFLIB_PATH=/path/to/MM-TSFlib (pinned at e789ce78) first")
    if "," in a.tsfn and {"unimodal", "timemmd"} & set(a.arms):
        sys.exit("unimodal/timemmd take one numeric expert; use --arms gmmts mogu with several")
    runs_dir = Path(a.runs_dir).resolve()

    domains = []
    for d in a.domains:
        path = Path(mm_path) / "data" / d / DOMAINS[d][0]
        if path.exists():
            domains.append(d)
        else:
            print("!! SKIPPING {}: {} not found{}".format(
                d, path, " (extract NewYork_AQI_Day.rar first)" if d == "Environment" else ""))

    config = REPO / "all_experts_config.csv"
    if not config.exists() and "gmmts" in a.arms:
        import pandas as pd
        from make_online_experts_config import build_rows
        pd.DataFrame(build_rows()).to_csv(config, index=False)
        print("wrote", config)

    jobs = []
    for d in domains:
        for pl in (a.horizons or DOMAINS[d][4]):
            for seed in a.seeds:
                for arm in a.arms:
                    for pw in (a.timemmd_pw if arm == "timemmd" else [0.0]):
                        jobs.append(Job(arm, pw, d, pl, seed, runs_dir))
    todo = sorted([j for j in jobs if not j.done()], key=lambda j: -j.cost)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    slots = [g for g in gpus for _ in range(a.jobs_per_gpu)]
    print("{} jobs, {} already done, {} to run on {} worker slot(s) {}; results in {}".format(
        len(jobs), len(jobs) - len(todo), len(todo), len(slots), slots, runs_dir))
    if a.dry_run:
        for j in todo[:3]:
            print("  e.g.", " ".join(shlex.quote(c) for c in build_cmd(j, a, mm_path, config)))
        return 0

    provenance = dict(code_commit=git_rev(REPO), mmtsflib_commit=git_rev(mm_path))
    q = queue.Queue()
    for j in todo:
        q.put(j)
    lock = threading.Lock()
    state = dict(done=0, failed=[], cost_done=0.0, secs=0.0, t0=time.time())
    total_cost = sum(j.cost for j in todo) or 1.0

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            ok, secs, mse = run_job(job, gpu, a, mm_path, config, provenance)
            with lock:
                state["done"] += 1
                state["cost_done"] += job.cost
                state["secs"] += secs
                if not ok:
                    state["failed"].append(job)
                wall = time.time() - state["t0"]
                eta = wall / state["cost_done"] * (total_cost - state["cost_done"])
                print("[{}/{}] {:<17} {:<13} pl={:<3} seed={} gpu={}: {} {}  ({:.0f}s)  "
                      "elapsed {:.0f} min, ETA ~{:.0f} min".format(
                          state["done"], len(todo), job.label, job.domain, job.pl, job.seed, gpu,
                          "ok" if ok else "FAILED (see {}/log.txt)".format(job.dir),
                          "mse={:.4f}".format(mse) if ok else "", secs, wall / 60, eta / 60),
                      flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if state["failed"]:
        print("\n{} job(s) failed -- rerun the same command to retry only those:".format(
            len(state["failed"])))
        for j in state["failed"]:
            print("  ", j.dir / "log.txt")
    import collect_table1_matrix
    collect_table1_matrix.main(["--runs_dir", str(runs_dir)])
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

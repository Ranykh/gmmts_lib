#!/usr/bin/env python3
"""Repair result folders that got nested by `mv` into an existing directory.

    python scripts/repro/repair_nested_results.py            # dry run, changes nothing
    python scripts/repro/repair_nested_results.py --apply

WHAT WENT WRONG
---------------
run_mmmogu_sweep.sh renamed each finished run with

    mv "$RESULTS_DIR/$setting" "$RESULTS_DIR/$final"

`mv src dst` moves src INSIDE dst when dst is an existing directory. So re-running
a config that had already completed (without SKIP_DONE=1) produced

    <final>/metrics.npy            <- the OLD run
    <final>/<setting>/metrics.npy  <- the NEW run

The collector only scans the top level, so it silently kept reporting the old
numbers, and anything the new run added -- gate_weights.npy in particular --
was invisible.

The sweep script now clears the target before renaming, so this cannot recur.
This script cleans up the folders already affected.

WHICH COPY WINS
---------------
The inner folder is the later run by construction: it only exists because a
second run completed and was moved on top of the first. It is also the only one
that can contain gate_weights.npy. So the inner copy is promoted and the outer
files it replaces are overwritten.

Anything in the outer folder that the inner does not have is KEPT rather than
deleted -- promoting a run should not lose artefacts that the rerun happened not
to write.
"""
import argparse
import os
import shutil
import sys


def find_nested(root):
    """Result folders that contain a sub-directory holding its own metrics.npy."""
    out = []
    if not os.path.isdir(root):
        sys.exit(f"no such directory: {root}")
    for name in sorted(os.listdir(root)):
        outer = os.path.join(root, name)
        if not os.path.isdir(outer):
            continue
        for sub in sorted(os.listdir(outer)):
            inner = os.path.join(outer, sub)
            if os.path.isdir(inner) and os.path.exists(
                    os.path.join(inner, "metrics.npy")):
                out.append((outer, inner))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.abspath(
        os.path.join(here, "..", "..", "test_online_gating_results")))
    ap.add_argument("--apply", action="store_true",
                    help="actually move files. Without this, nothing is changed.")
    args = ap.parse_args()

    nested = find_nested(args.results)
    if not nested:
        print(f"no nested result folders under {args.results} -- nothing to do")
        return 0

    print(f"{len(nested)} nested folder(s) found under {args.results}\n")
    promoted = 0
    for outer, inner in nested:
        inner_files = sorted(os.listdir(inner))
        outer_files = sorted(f for f in os.listdir(outer)
                             if os.path.isfile(os.path.join(outer, f)))
        kept = [f for f in outer_files if f not in inner_files]
        print(f"  {os.path.basename(outer)[:78]}")
        print(f"      inner (newer, promoted): {', '.join(inner_files)}")
        if kept:
            print(f"      outer-only, kept as-is:  {', '.join(kept)}")
        if args.apply:
            for f in inner_files:
                shutil.move(os.path.join(inner, f), os.path.join(outer, f))
            os.rmdir(inner)
            promoted += 1
        print()

    if args.apply:
        print(f"promoted {promoted} folder(s). Re-run collect_gmmts.py.")
    else:
        print("DRY RUN -- nothing changed. Re-run with --apply to do it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Copy the result files of a fresh run (outputs/) into paper/data/, the layout make_numbers.py reads.

    python paper/scripts/collect_results.py [--outputs outputs]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["config", "eval", "probe", "kernel", "mech", "transplant", "baseline", "posterior",
         "r7_misspec", "r7_interchange", "r7_bank", "r8_misspec_ci", "r8_pooled", "r8_das",
         "r9_ablate", "r9_das_controls"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    out = Path(args.outputs)
    data = ROOT / "paper" / "data"
    n = 0
    for run_dir in sorted((out / "hawkes_main").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "config.json").exists():
            continue
        dest = data / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            src = run_dir / f"{name}.json"
            if src.exists():
                shutil.copy2(src, dest / src.name)
                n += 1
    fim = out / "fimpp" / "fimpp.json"
    if fim.exists():
        (data / "fimpp").mkdir(parents=True, exist_ok=True)
        shutil.copy2(fim, data / "fimpp" / "fimpp.json")
        n += 1
    print(f"copied {n} result files into {data}")


if __name__ == "__main__":
    main()

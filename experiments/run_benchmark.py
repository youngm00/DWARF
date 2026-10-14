"""Benchmark experiment: DWARF (default CBS) on the 15 datasets x 30 seeds x 5 folds.

    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --datasets iris wine --seeds 1 --workers 2

Results: results/benchmark_<mode>.csv (one row per dataset, seed and fold).
"""
import argparse
import os

import pandas as pd

import runner


def main():
    cfg = runner.load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=runner.D.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=runner.load_seeds(cfg))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(1, cfg["experiment"]["n_folds"] + 1)))
    ap.add_argument("--mode", default=cfg["dwarf"]["candidate_mode"], choices=["midpoint", "cb", "cbs"])
    ap.add_argument("--fitness", default=cfg["dwarf"]["fitness_metric"],
                    choices=["accuracy", "balanced_accuracy", "macro_f1", "mcc", "gmean"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(runner.ROOT, "results", "benchmark_%s.csv" % args.mode)
    units = [(d, s, f, args.mode, args.fitness, cfg) for d in args.datasets for s in args.seeds for f in args.folds]
    runner.run(units, out, args.workers)

    d = pd.read_csv(out)
    d = d[(d.status == "ok") & (d["mode"] == args.mode) & d.dataset.isin(args.datasets)]
    print("\nmean over seeds and folds (%s):" % args.mode)
    print(d.groupby("dataset")[["test_accuracy", "test_macro_f1", "total_nodes"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()

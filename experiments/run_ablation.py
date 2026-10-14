"""Candidate-generation ablation: DWARF-All (midpoint) vs DWARF-CB vs DWARF-CBS.

    python experiments/run_ablation.py
    python experiments/run_ablation.py --datasets iris wine --seeds 1 --workers 2

Every other setting is the default of config.yaml. Results: results/ablation.csv and
results/ablation_summary.csv (mean over seeds and folds per dataset and mode).
"""
import argparse
import os

import pandas as pd

import runner

MODES = ["midpoint", "cb", "cbs"]
LABEL = {"midpoint": "DWARF-All", "cb": "DWARF-CB", "cbs": "DWARF-CBS"}


def main():
    cfg = runner.load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=runner.D.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=runner.load_seeds(cfg))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(1, cfg["experiment"]["n_folds"] + 1)))
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=os.path.join(runner.ROOT, "results", "ablation.csv"))
    args = ap.parse_args()

    units = [(d, s, f, m, None, cfg) for m in args.modes for d in args.datasets
             for s in args.seeds for f in args.folds]
    runner.run(units, args.out, args.workers)

    d = pd.read_csv(args.out)
    d = d[(d.status == "ok") & d.dataset.isin(args.datasets) & d["mode"].isin(args.modes)]
    s = d.groupby(["dataset", "mode"])[["test_accuracy", "test_macro_f1", "total_nodes", "n_tree_fits",
                                        "runtime_sec"]].mean().reset_index()
    s["mode"] = s["mode"].map(LABEL)
    s.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(s.pivot(index="dataset", columns="mode", values="test_macro_f1").round(4).to_string())


if __name__ == "__main__":
    main()

"""Shared driver: run DWARF over (dataset, seed, fold) units in parallel and append rows to a CSV."""
import contextlib
import io
import logging
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import datasets as D  # noqa: E402


def load_config(path=None):
    with open(path or os.path.join(HERE, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seeds(cfg):
    with open(os.path.join(ROOT, cfg["experiment"]["seeds_file"]), encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def build_config(cfg, name, seed, mode=None, fitness_metric=None):
    types, drop, default_return = D.dataset_meta(name)
    c = {}
    c.update(cfg["dwarf"])
    c.update(cfg["ga"])
    c.update({"algorithm": cfg["tree"]["algorithm"], "max_depth": cfg["tree"]["max_depth"],
              "enableParallelism": False, "enableGBM": False,
              "feature_value_map": {}, "default_return": default_return,
              "column_types": types, "drop_cols": drop, "seed": seed})
    if mode:
        c["candidate_mode"] = mode
    if fitness_metric:
        c["fitness_metric"] = fitness_metric
    return c


def run_unit(args):
    name, seed, fold, mode, fitness_metric, cfg = args
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.chdir(ROOT)  # chefboost writes its rule files to ./outputs/rules
    from dwarf import run_fold
    t0 = time.time()
    try:
        config = build_config(cfg, name, seed, mode, fitness_metric)
        with contextlib.redirect_stdout(io.StringIO()):  # chefboost is verbose
            row = run_fold(config, D.split_path(name, fold, "train"), D.split_path(name, fold, "test"), fold,
                           rule_name="%s_%s_s%d_f%d" % (name, config["candidate_mode"], seed, fold))
        row.update({"dataset": name, "mode": config["candidate_mode"], "seed": seed,
                    "fitness_metric": config["fitness_metric"], "status": "ok", "error": ""})
    except Exception as exc:
        row = {"dataset": name, "mode": mode, "seed": seed, "fold": fold, "status": "failed",
               "error": "%s: %s" % (type(exc).__name__, exc), "traceback": traceback.format_exc()[-800:],
               "runtime_sec": time.time() - t0}
    return row


def _append(row, out_csv):
    df = pd.DataFrame([row])
    if not os.path.exists(out_csv):
        df.to_csv(out_csv, index=False)
        return
    cols = list(pd.read_csv(out_csv, nrows=0).columns)
    if set(df.columns) <= set(cols):
        df.reindex(columns=cols).to_csv(out_csv, mode="a", header=False, index=False)
    else:
        pd.concat([pd.read_csv(out_csv), df], ignore_index=True).to_csv(out_csv, index=False)


def run(units, out_csv, workers):
    """units: (dataset, seed, fold, mode, fitness_metric, cfg). Units already 'ok' in out_csv are skipped."""
    done = set()
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv)
        d = d[d.status == "ok"]
        done = set(zip(d.dataset, d["mode"], d.seed, d.fold))
    todo = [u for u in units if (u[0], u[3], u[1], u[2]) not in done]
    print("%d units to run (%d already done), %d workers -> %s" % (len(todo), len(units) - len(todo), workers, out_csv),
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_unit, u) for u in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            _append(row, out_csv)
            res = ("test acc %.4f  nodes %s" % (row["test_accuracy"], row["total_nodes"])
                   if row["status"] == "ok" else row["error"])
            print("[%d/%d] %s %s seed %s fold %s: %s" % (k, len(todo), row["dataset"], row.get("mode"),
                                                        row["seed"], row["fold"], res), flush=True)

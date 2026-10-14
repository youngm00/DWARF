# DWARF

Youngmin Kim, Chul-woo Lee, Sungbum Jun\
Department of Industrial and Systems Engineering, Dongguk University

DWARF is an evolutionary discretization framework for accurate and interpretable
depth-limited decision trees. It builds such trees by searching for a good discretization of the numerical
attributes. A random-forest pre-selection keeps the most important attributes; for each
selected numerical attribute a pool of candidate cut points is generated (all midpoints,
class boundaries, or class boundaries that pass a look-ahead stability filter); a genetic
algorithm then searches over how many intervals each attribute gets and where the cuts
lie, scoring every discretization by the training accuracy of the depth-limited
multi-way C4.5 tree grown on it. The best discretization yields a shallow tree whose
splits are readable interval conditions.

## Installation

Python 3.10 was used for the paper.

```bash
pip install -r requirements.txt
```

The repository contains a modified copy of [chefboost](https://github.com/serengil/chefboost)
(`chefboost/`, MIT License). Do not install chefboost from PyPI; the bundled copy is required
to reproduce the results (see `chefboost/MODIFICATIONS.md`).

## Quick start

Run DWARF-CBS with the default parameters on one dataset, one seed and all five folds:

```bash
python experiments/run_benchmark.py --datasets iris --seeds 1 --workers 5
```

Or call it from Python on a single split (run from the repository root; chefboost writes
the tree as a Python rule file under `outputs/rules/`):

```python
import sys; sys.path.insert(0, "experiments")
import runner, datasets
from dwarf import run_fold

cfg = runner.load_config()
config = runner.build_config(cfg, "iris", seed=1)
result = run_fold(config, datasets.split_path("iris", 1, "train"),
                  datasets.split_path("iris", 1, "test"), fold=1)
print(result["test_accuracy"], result["total_nodes"], result["rule_file"])
```

## Running the experiments

| Experiment | Command | Output |
|---|---|---|
| Benchmark (15 datasets x 30 seeds x 5 folds, DWARF-CBS) | `python experiments/run_benchmark.py` | `results/benchmark_cbs.csv` |
| Candidate generation: All vs CB vs CBS | `python experiments/run_ablation.py` | `results/ablation.csv`, `results/ablation_summary.csv` |

Common options: `--datasets`, `--seeds`, `--folds`, `--workers`. `run_benchmark.py` also
takes `--mode {midpoint,cb,cbs}` and `--fitness {accuracy,balanced_accuracy,macro_f1,mcc,gmean}`.
Finished (dataset, mode, seed, fold) rows are skipped when a run is restarted.

Default parameters (`experiments/config.yaml`):

| Parameter | Value |
|---|---|
| Candidate mode | CBS |
| tau (cumulative RF importance of the pre-selection) | 0.7 |
| f_min (minimum number of pre-selected attributes) | 5 |
| b_max (maximum intervals per attribute) | 3 |
| L (look-ahead window of CBS) | 4 |
| Population / generations | 100 / 20 |
| Survival (elite) / mutation / crossover / random rates | 0.1 / 0.3 / 0.4 / 0.2 |
| Tree | C4.5, maximum depth 3 |
| Fitness | training accuracy (`balanced_accuracy`, `macro_f1`, `mcc`, `gmean` available) |

Each run seeds Python's and NumPy's random generators with `seed + fold`.

## Code layout

```
dwarf/
  candidates.py         candidate cut points (All / CB / CBS)
  feature_selection.py  cumulative-importance pre-selection
  ga.py                 genes, chromosomes, mutation, crossover, evolution loop
  tree.py               depth-limited multi-way C4.5 (chefboost), rule-file evaluation, tree size
  pipeline.py           run_fold: one complete DWARF run on a train/test split
experiments/            config.yaml, dataset metadata, benchmark and ablation drivers
chefboost/              modified chefboost 0.0.18 (MIT)
data/                   processed datasets, splits, split script
seeds.txt               the 30 seeds
```

## Data

| Dataset | Source | Instances | Classes |
|---|---|---|---|
| Iris | https://archive.ics.uci.edu/dataset/53/iris | 150 | 3 |
| Wine | https://archive.ics.uci.edu/dataset/109/wine | 178 | 3 |
| Heart (Cleveland) | https://archive.ics.uci.edu/dataset/45/heart+disease | 303 | 5 |
| Parkinsons | https://archive.ics.uci.edu/dataset/174/parkinsons | 195 | 2 |
| Breast Cancer Wisconsin (Original) | https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original | 699 | 2 |
| Blood Transfusion | https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center | 748 | 2 |
| Fertility | https://archive.ics.uci.edu/dataset/244/fertility | 100 | 2 |
| Hepatitis | https://archive.ics.uci.edu/dataset/46/hepatitis | 155 | 2 |
| Thoracic Surgery | https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data | 470 | 2 |
| Bank Marketing | https://archive.ics.uci.edu/dataset/222/bank+marketing | 45,211 | 2 |
| Census Income (Adult) | https://archive.ics.uci.edu/dataset/20/census+income | 32,561 | 2 |
| Ionosphere | https://archive.ics.uci.edu/dataset/52/ionosphere | 351 | 2 |
| Palmer Penguins | https://allisonhorst.github.io/palmerpenguins/ | 344 | 3 |
| CNC | https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill | 32,048 | 2 |
| Faulty Steel Plates | https://archive.ics.uci.edu/dataset/198/steel+plates+faults | 1,941 | 7 |

* `data/processed/<name>.csv` is the file used in the experiments: one row per instance, the
  class in the last column `Decision`. Census Income is the `adult.data` part (32,561 rows).
* `data/splits/<name>/<name>_fold_<i>_{train,test}.csv` are the stratified 5-fold splits used
  for every model in the paper. `python data/make_splits.py` recreates them from
  `data/processed/` (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
* Missing values: hepatitis was imputed after splitting, in each fold file separately
  (attributes with at most two distinct values: mode; other numerical attributes: median),
  which `make_splits.py` reproduces. For the other datasets, rows with missing values
  (breast cancer 16, heart 6, penguins 11) are dropped when a fold is loaded.
* Faulty Steel Plates: `processed/steel_plates_faults.csv` keeps the seven one-hot fault
  columns from which `Decision` was built; they are removed before training
  (`DROP_COLS` in `experiments/datasets.py`).
* `seeds.txt` lists the 30 seeds (1, 11, ..., 291). Seeds are 10 apart because each run uses
  `seed + fold`, so no two (seed, fold) pairs share a random stream.

The industrial vehicle factory (IVF) dataset cannot be released due to a
confidentiality agreement with the manufacturer.

## License

MIT (see `LICENSE`). The bundled chefboost is MIT-licensed by its author (`chefboost/LICENSE`).

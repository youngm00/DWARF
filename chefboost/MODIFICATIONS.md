# Modified chefboost

This directory is a modified copy of [chefboost](https://github.com/serengil/chefboost)
0.0.18 by Sefik Ilkin Serengil, distributed under the MIT License (see `LICENSE`).
DWARF needs this copy: the PyPI package produces different trees and a different random
stream. The complete difference to the PyPI release 0.0.18 is in `CHANGES_vs_0.0.18.diff`.

## Changes that affect the output

* **Branches for unseen values** (`feature_value_map`, `default_return`): when a node splits
  on a categorical (or discretized) attribute, every value listed for that attribute in
  `config["feature_value_map"]` gets a branch. A value with no training instance at that node
  gets a leaf returning `config["default_return"]`. DWARF sets the map from the discretized
  training data; the experiment scripts set `default_return` to the majority class of the
  dataset's first training fold (`experiments/datasets.py`), as in the paper's runs.
* **Deterministic leaf ids**: chefboost created leaf ids with `uuid.uuid1()`, which consumes
  the global `random` module (CPython uses `random.getrandbits` for the clock sequence). This
  disturbed the random stream of the genetic search. Leaf ids now come from a counter
  (`Training._new_leaf_id`); they never appear in the rule files.
* **Training predictions in the evaluation result**: `evaluate` also returns the actual and
  predicted training labels, so that DWARF can use fitness measures other than accuracy.
* **Optional leaf labelling** `config["leaf_label"] = "prior"` (off by default, not used by
  DWARF's defaults).
* With `enableParallelism`, the JSON rule file is named after the model instead of `rules.json`.

## Speed-only changes (trees and predictions unchanged)

* Rule lines are buffered in memory and written once per file (`commons/rulebuffer.py`).
* Gain / gain-ratio computations for ID3 and C4.5 use numpy contingency tables and cumulative
  sums instead of per-threshold DataFrame filtering.
* Removed a forced `gc.collect()` per branch and a process-table scan that only matters with
  `enableParallelism`.
* Training-set prediction iterates over a numpy array instead of `DataFrame.iterrows()`;
  debug log strings are only built when debug logging is on.
* `commons/module.py` loads a rule file directly from its path.

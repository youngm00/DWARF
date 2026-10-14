"""One DWARF run on one train/test split."""
import os
import random
import time

import numpy as np
import pandas as pd

from . import tree
from .feature_selection import select_features
from .ga import evolve


def run_fold(config, train_path, test_path, fold, rule_name=None):
    """Run DWARF on one fold and return a dict of results.

    config keys (see experiments/config.yaml):
      candidate_mode ('midpoint' | 'cb' | 'cbs'), fi_threshold (tau), top_features (f_min),
      max_disc (b_max), cbs_lookahead (L), population_size, generations, mutation_rate,
      crossover_rate, random_rate, survival_rate, fitness_metric, fitness_cache, seed,
      column_types {column: 'Numerical' | 'Categorical' | 'Decision'}, drop_cols,
      and the chefboost keys algorithm, max_depth, enableParallelism, enableGBM,
      feature_value_map, default_return.
    The random generators are seeded with seed + fold.
    """
    seed = config.get('seed', 42)
    if seed is not None:
        random.seed(seed + fold)
        np.random.seed(seed + fold)
    drop_cols = config.get('drop_cols') or []
    start = time.time()

    df_train = pd.read_csv(train_path)
    df_train['Decision'] = df_train['Decision'].astype(str)
    df_train = df_train.dropna()
    df_test = pd.read_csv(test_path)
    df_test['Decision'] = df_test['Decision'].astype(str)
    df_test = df_test.dropna()
    if drop_cols:
        df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
        df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    df_train, df_test, cat_cols = select_features(
        df_train, df_test, config['column_types'],
        top_k=config['top_features'], threshold=config.get('fi_threshold', 0.7))
    n_selected = df_train.shape[1] - 1
    n_numeric = sum(1 for c in df_train.columns
                    if c != 'Decision' and config['column_types'].get(c) == 'Numerical')

    best, best_train_acc, best_fitness, n_calls, n_fits = evolve(df_train, cat_cols, config)

    # refit the best chromosome under its own name and evaluate the saved tree on the test fold
    name = rule_name or 'dwarf_%s_fold%d_%d' % (config.get('candidate_mode', 'midpoint'), fold, os.getpid())
    best.save_model(df_train, name)
    stats = tree.tree_stats(tree.rule_path(name)) or {}
    _, _, train_f1 = best.train_metrics_macro(df_train)
    test_acc, test_p, test_r, test_f1 = tree.evaluate_rule_file(test_path, name, config['column_types'])

    return {
        'fold': fold,
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1,
        'test_macro_precision': test_p,
        'test_macro_recall': test_r,
        'train_accuracy': float(best_train_acc) / 100.0,
        'train_macro_f1': float(train_f1) / 100.0,
        'best_fitness': best_fitness,
        'decision_nodes': stats.get('decision_nodes'),
        'leaves': stats.get('leaves'),
        'total_nodes': stats.get('total_nodes'),
        'depth': stats.get('depth'),
        'max_branching': stats.get('max_branching'),
        'n_selected_features': n_selected,
        'n_selected_numeric': n_numeric,
        'n_fitness_calls': n_calls,
        'n_tree_fits': n_fits,
        'runtime_sec': time.time() - start,
        'rule_file': tree.rule_path(name),
    }

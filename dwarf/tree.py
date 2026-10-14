"""Depth-limited multi-way decision tree (C4.5) on discretized data.

Trees are grown with the bundled ``chefboost`` (C4.5, ``max_depth``). A discretized
attribute with k intervals becomes a k-way split; an attribute left continuous is split
by C4.5 with a binary threshold. chefboost writes each tree as a Python rule file
(``outputs/rules/<name>.py``, relative to the working directory); this module evaluates
those rule files and measures their size.
"""
import ast
import importlib.util
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from chefboost import Chefboost as chef

RULE_DIR = os.path.join('outputs', 'rules')
_RULE_LABEL_RE = re.compile(r"obj\[(\d+)\]\s*==\s*'(?:Less|Greater)_([-\d.eE+]+)'")


def fit(df_disc, config, target, name):
    """Fit a C4.5 tree on a discretized frame. chefboost stores it as RULE_DIR/<name>.py."""
    return chef.fit(df_disc, config, target_label=target, name=name)


def predict(model, row):
    return chef.predict(model, row)


def rule_path(name):
    return os.path.join(RULE_DIR, name + '.py')


# --------------------------------------------------------------------------- rule file evaluation

def extract_features_from_model(py_file):
    """Attribute names in the order the rule file indexes them (obj[i])."""
    with open(py_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    feats = re.findall(r"obj\[\d+\]:\s*([^,\]#\n]+)", first_line)
    return [x.strip() for x in feats]


def parse_rule_cutpoints(py_file):
    """{obj index: sorted cut points} read back from the interval labels in the rule file."""
    src = open(py_file, "r", encoding="utf-8").read()
    cuts = {}
    for idx, val in _RULE_LABEL_RE.findall(src):
        cuts.setdefault(int(idx), set()).add(float(val))
    return {k: sorted(v) for k, v in cuts.items()}


def discretize_like_gda(series, cutpoints):
    """Same intervals and labels as ga.discrete_df."""
    bins = [float("-inf")] + list(cutpoints) + [float("inf")]
    labels = [f"Less_{c}" for c in cutpoints] + [f"Greater_{cutpoints[-1]}"]
    return pd.cut(series, bins=bins, labels=labels).astype("object")


def build_rule_input(df, features, cutpoints, column_dict):
    """Map raw data onto the interval labels used by the rule file, so the rule runs unchanged."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"columns referenced by the rule are missing: {missing}")
    cols = {}
    for i, f in enumerate(features):
        if i in cutpoints:                              # discretized numerical attribute
            cols[f] = discretize_like_gda(df[f].astype(float), cutpoints[i])
        elif column_dict.get(f) == "Categorical":
            cols[f] = df[f].astype(str)
        else:                                           # numerical attribute left continuous
            cols[f] = df[f].astype(float)
    return pd.DataFrame(cols, index=df.index)


def evaluate_rule_file(data_path, rule_name, column_dict):
    """Apply a saved rule file to a CSV and return (accuracy, macro P, macro R, macro F1)."""
    df = pd.read_csv(data_path)
    df['Decision'] = df['Decision'].astype(str)
    df = df.dropna()

    path = rule_path(rule_name)
    features = extract_features_from_model(path)
    X_eval = build_rule_input(df, features, parse_rule_cutpoints(path), column_dict)

    spec = importlib.util.spec_from_file_location(f"saved_model_{os.getpid()}", path)
    saved_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saved_model)
    predictions = X_eval.apply(lambda row: saved_model.findDecision(row.tolist()), axis=1)

    y_true = df['Decision'].astype(str)
    y_pred = predictions.astype(str)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


# --------------------------------------------------------------------------- tree size

def _chain(node):
    """Split an if/elif chain into [(condition, body)] and the final else body."""
    branches = []
    cur = node
    while True:
        branches.append((cur.test, cur.body))
        if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
            continue
        return branches, cur.orelse


def tree_stats(path):
    """Structure of a chefboost rule file.

    decision_nodes : internal nodes (one if/elif chain = one node)
    leaves         : reachable leaves (chefboost's unreachable fallback ``else: return`` excluded)
    total_nodes    : decision_nodes + leaves
    max_branching  : largest number of branches of one node
    depth          : tree depth
    Returns None if the file is missing or cannot be parsed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return None
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "findDecision"), None)
    if fn is None:
        return None

    stats = {"decision_nodes": 0, "leaves": 0, "max_branching": 0, "depth": 0}

    def walk(body, depth):
        stats["depth"] = max(stats["depth"], depth)
        ifs = [s for s in body if isinstance(s, ast.If)]
        if not ifs:
            if any(isinstance(s, ast.Return) for s in body):
                stats["leaves"] += 1
            return
        for node in ifs:
            branches, final_else = _chain(node)
            stats["decision_nodes"] += 1
            stats["max_branching"] = max(stats["max_branching"], len(branches))
            for _, sub in branches:
                walk(sub, depth + 1)
            if final_else and any(isinstance(s, ast.If) for s in final_else):
                walk(final_else, depth + 1)

    walk(fn.body, 0)
    stats["total_nodes"] = stats["decision_nodes"] + stats["leaves"]
    return stats

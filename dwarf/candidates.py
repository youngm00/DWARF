"""Candidate cut-point generation for one numerical attribute.

Three modes are supported:

* ``midpoint`` (DWARF-All): midpoints of all adjacent distinct values.
* ``cb``       (DWARF-CB) : midpoints where the class label changes (class boundaries).
* ``cbs``      (DWARF-CBS): class boundaries that pass a look-ahead majority
  stability filter of window ``L``.
"""
from typing import List

import numpy as np

CBS_LOOKAHEAD = 4  # default look-ahead window L for the CBS stability filter


def build_candidate_bounds(values, labels, mode: str = 'midpoint', lookahead: int = CBS_LOOKAHEAD) -> List[float]:
    """Return the sorted list of candidate cut points for one attribute.

    values : attribute values, labels : class labels in the same order.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return []

    if mode == 'midpoint':
        uniq = np.unique(values)
        if len(uniq) < 2:
            return []
        return ((uniq[:-1] + uniq[1:]) / 2.0).tolist()

    labels = np.asarray(labels)

    if mode == 'cb':
        # sort by value (ties broken by label), keep midpoints where the label changes
        order = np.lexsort((labels, values))
        sv, sl = values[order], labels[order]
        bounds = set()
        for i in range(n - 1):
            if sl[i] != sl[i + 1]:
                bounds.add(float((sv[i] + sv[i + 1]) / 2.0))
        # drop candidates that coincide with an observed value
        uniq = set(np.unique(values).tolist())
        return sorted(b for b in bounds if b not in uniq)

    if mode == 'cbs':
        # class boundary + look-ahead majority stability filter
        order = np.argsort(values, kind='mergesort')
        sv, sl = values[order], labels[order]
        bounds = []
        for i in range(n - 1):
            if sv[i] == sv[i + 1]:
                continue
            if sl[i] != sl[i + 1]:
                window = sl[i + 1: min(i + 1 + lookahead, n)]
                if np.sum(window == sl[i + 1]) >= (len(window) / 2):
                    bounds.append(float((sv[i] + sv[i + 1]) / 2.0))
        return sorted(set(bounds))

    raise ValueError(f"unknown candidate mode: {mode!r} (use 'midpoint' | 'cb' | 'cbs')")

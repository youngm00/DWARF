"""Recreate the stratified 5-fold splits in data/splits/ from data/processed/.

    python data/make_splits.py                 # writes data/splits/<name>/<name>_fold_<i>_{train,test}.csv
    python data/make_splits.py --out /tmp/x    # write elsewhere (e.g. to compare with the shipped splits)

StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the processed file
(target column 'Decision', rows with a missing target dropped).

hepatitis: after splitting, missing values are filled in every fold file separately
(train and test each with their own statistics): attributes with at most two distinct
values get the mode, other numerical attributes the median, non-numerical attributes the
mode. This is how the shipped hepatitis splits were produced.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
IMPUTED = {"hepatitis"}


def impute(df):
    df = df.replace('?', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 2:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df.dropna()


def split_dataset(name, src_dir, out_dir):
    df = pd.read_csv(os.path.join(src_dir, name + ".csv"))
    df = df[[c for c in df.columns if c != "Decision"] + ["Decision"]]
    df = df.dropna(subset=["Decision"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    for i, (tr, te) in enumerate(skf.split(df.drop(columns=["Decision"]), df["Decision"]), 1):
        for part, idx in (("train", tr), ("test", te)):
            fold = df.iloc[idx]
            if name in IMPUTED:
                fold = impute(fold.copy())
            fold.to_csv(os.path.join(out_dir, name, "%s_fold_%d_%s.csv" % (name, i, part)), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "processed"))
    ap.add_argument("--out", default=os.path.join(HERE, "splits"))
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    names = args.datasets or sorted(f[:-4] for f in os.listdir(args.src) if f.endswith(".csv"))
    for name in names:
        split_dataset(name, args.src, args.out)
        print("split", name)


if __name__ == "__main__":
    main()

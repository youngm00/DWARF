"""Attribute pre-selection by cumulative random-forest importance.

Attributes are ranked by RF importance (one-hot columns of a categorical attribute are
summed back into that attribute). The smallest prefix whose cumulative importance reaches
``threshold`` (tau) is kept, but never fewer than ``top_k`` (f_min) attributes.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def onehot_and_group_importance(df_train, df_test, target_col, top_k, sep="=", threshold=0.7):
    """Select attributes on the training fold and apply the same selection to the test fold.

    Returns (df_train_selected, df_test_selected, grouped_importance, onehot_importance, n_selected).
    """
    X_train_raw = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].astype(str)

    X_test_raw = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(str)

    X_train_oh = pd.get_dummies(X_train_raw, prefix_sep='=')

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_oh, y_train)

    fi = pd.Series(rf.feature_importances_, index=X_train_oh.columns)

    # one-hot columns -> original attribute
    def base_feature(col):
        return col.split(sep)[0]

    grouped_fi = fi.groupby(fi.index.map(base_feature)).sum().sort_values(ascending=False)

    cum = grouped_fi.cumsum()
    if threshold >= 1.0:
        # a floating-point cumulative sum may never reach exactly 1.0 -> keep every attribute
        selected_cols_names = grouped_fi.index
    else:
        selected_cols_names = grouped_fi.index[:(cum >= threshold).argmax() + 1]

    if len(selected_cols_names) <= top_k:
        top_base = grouped_fi.head(top_k).index.tolist()
        num_top_k = top_k
    else:
        top_base = list(selected_cols_names)
        num_top_k = len(selected_cols_names)

    missing = [c for c in top_base if c not in X_train_raw.columns]
    if missing:
        raise KeyError(f"Selected base features not found in raw columns: {missing}")

    df_train_final = pd.concat([X_train_raw[top_base], y_train], axis=1)
    df_test_final = pd.concat([X_test_raw[top_base], y_test], axis=1)

    return df_train_final, df_test_final, grouped_fi, fi, num_top_k


def select_features(df_train, df_test, column_types, top_k, threshold=0.7, target_col='Decision'):
    """Cast columns to their declared type, run the pre-selection and return the
    selected frames together with the list of selected categorical attributes."""
    for c in df_train.columns:
        if column_types.get(c) == 'Categorical':
            df_train[c] = df_train[c].astype(str)
            df_test[c] = df_test[c].astype(str)
        elif column_types.get(c) == 'Numerical':
            df_train[c] = df_train[c].astype(float)
            df_test[c] = df_test[c].astype(float)

    df_train, df_test, grouped_fi, _, count_col = onehot_and_group_importance(
        df_train=df_train, df_test=df_test, target_col=target_col,
        top_k=top_k, sep="=", threshold=threshold)

    cat_cols = [c for c in grouped_fi.head(count_col).index.tolist()
                if column_types.get(c) == 'Categorical']
    return df_train, df_test, cat_cols

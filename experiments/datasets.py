"""The 15 benchmark datasets and their column metadata."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "splits")

DATASETS = [
    "bank_marketing", "blood_transfusion", "breast_cancer_wisconsin", "CNC", "census_income",
    "fertility", "heart", "hepatitis", "ionosphere", "iris", "parkinsons", "penguins",
    "steel_plates_faults", "thoracic_surgery_data", "wine",
]

# Column types are inferred from the dtype except for heart, whose integer-coded
# categorical attributes are declared explicitly.
COLUMN_TYPES = {
    "heart": {
        "age": "Numerical", "sex": "Categorical", "cp": "Categorical", "trestbps": "Numerical",
        "chol": "Numerical", "fbs": "Categorical", "restecg": "Categorical", "thalach": "Numerical",
        "exang": "Categorical", "oldpeak": "Numerical", "slope": "Categorical", "ca": "Numerical",
        "thal": "Categorical", "Decision": "Decision",
    },
}

# steel_plates_faults ships the class as seven one-hot columns next to Decision;
# they are removed before training.
DROP_COLS = {
    "steel_plates_faults": ["Pastry", "Z_Scratch", "K_Scratch", "Stains",
                            "Dirtiness", "Bumps", "Other_Faults"],
}


def split_path(name, fold, part):
    return os.path.join(SPLIT_DIR, name, "%s_fold_%d_%s.csv" % (name, fold, part))


def infer_column_types(df, target="Decision"):
    types = {}
    for col in df.columns:
        if col == target:
            types[col] = "Decision"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "Numerical"
        else:
            types[col] = "Categorical"
    return types


def dataset_meta(name):
    """(column_types, drop_cols, default_return) of a dataset, read from its fold-1 training file."""
    df = pd.read_csv(split_path(name, 1, "train"))
    drop = DROP_COLS.get(name, [])
    df = df.drop(columns=[c for c in drop if c in df.columns])
    types = COLUMN_TYPES.get(name) or infer_column_types(df)
    return types, drop, str(df["Decision"].mode()[0])

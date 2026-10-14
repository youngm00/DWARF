import math
import numpy as np
import pandas as pd
from chefboost.training import Training
from chefboost.commons.logger import Logger

logger = Logger(module="chefboost/training/Preprocess.py")


def _entropy_from_counts(counts, instances: int) -> float:
    """Entropy from a vector of class counts (same result as Training.calculateEntropy)."""
    if instances == 0:
        return 0
    entropy = 0
    for idx in np.argsort(-counts, kind="stable"):
        c = int(counts[idx])
        if c == 0:
            break
        class_probability = c / instances
        entropy = entropy - class_probability * math.log(class_probability, 2)
    return entropy


def _fast_continuous_metrics(df, column_name, entropy, config, unique_values):
    """Gain / gain ratio of every threshold of a continuous attribute via cumulative class counts (ID3/C4.5)."""
    algorithm = config["algorithm"]
    vals = df[column_name].to_numpy()
    if vals.dtype.kind not in "iuf" or np.isnan(vals).any():
        return None

    dec_codes, dec_uniques = pd.factorize(df["Decision"], sort=False)
    if len(dec_uniques) == 0 or (dec_codes < 0).any():
        return None
    n_class = len(dec_uniques)
    n = len(vals)

    order = np.argsort(vals, kind="stable")
    vals_sorted = vals[order]
    dec_sorted = dec_codes[order]

    cum = np.zeros((n_class, n + 1), dtype=np.int64)
    for c in range(n_class):
        cum[c, 1:] = np.cumsum(dec_sorted == c)
    total_counts = cum[:, n]

    subset_gains = []
    subset_gainratios = []
    for i in range(0, len(unique_values) - 1):
        threshold = unique_values[i]
        k = int(np.searchsorted(vals_sorted, threshold, side="right"))

        subset1_rows = k
        subset2_rows = n - k
        subset1_probability = subset1_rows / n
        subset2_probability = subset2_rows / n

        e1 = _entropy_from_counts(cum[:, k], subset1_rows)
        e2 = _entropy_from_counts(total_counts - cum[:, k], subset2_rows)

        threshold_gain = entropy - subset1_probability * e1 - subset2_probability * e2
        subset_gains.append(threshold_gain)

        if algorithm == "C4.5":
            threshold_splitinfo = -subset1_probability * math.log(
                subset1_probability, 2
            ) - subset2_probability * math.log(subset2_probability, 2)
            subset_gainratios.append(threshold_gain / threshold_splitinfo)

    return subset_gains, subset_gainratios


def processContinuousFeatures(
    algorithm: str, df: pd.DataFrame, column_name: str, entropy: float, config: dict
) -> pd.DataFrame:
    """
    Find the best split point for numeric features
    Args:
        df (pd.DataFrame): (sub) training dataframe
        column_name (str): current column to process
        entropy (float): calculated entropy
        config (dict): training configuration
    Returns
        df (pd.DataFrame): dataframe with numeric columns updated
            to nominal (e.g. instead of continious age >40 or <=40)
    """
    # if True:
    if df[column_name].nunique() <= 20:
        unique_values = sorted(df[column_name].unique())
    else:
        unique_values = []

        df_mean = df[column_name].mean()
        df_std = df[column_name].std(ddof=0)
        df_min = df[column_name].min()
        df_max = df[column_name].max()

        unique_values.append(df[column_name].min())
        unique_values.append(df[column_name].max())
        unique_values.append(df[column_name].mean())

        scales = list(range(-3, +4, 1))
        for scale in scales:
            if df_mean + scale * df_std > df_min and df_mean + scale * df_std < df_max:
                unique_values.append(df_mean + scale * df_std)

        unique_values.sort()

    logger.debug(f"{column_name} -> {unique_values}")

    subset_gainratios = []
    subset_gains = []
    subset_ginis = []
    subset_red_stdevs = []
    subset_chi_squares = []

    if len(unique_values) == 1:
        winner_threshold = unique_values[0]
        df[column_name] = np.where(
            df[column_name] <= winner_threshold,
            "<=" + str(winner_threshold),
            ">" + str(winner_threshold),
        )
        return df

    _fast = None
    if algorithm in ["ID3", "C4.5"]:
        _fast = _fast_continuous_metrics(df, column_name, entropy, config, unique_values)

    if _fast is not None:
        subset_gains, subset_gainratios = _fast

    for i in range(0, len(unique_values) - 1):
        if _fast is not None:
            break

        threshold = unique_values[i]

        subset1 = df[df[column_name] <= threshold]
        subset2 = df[df[column_name] > threshold]

        subset1_rows = subset1.shape[0]
        subset2_rows = subset2.shape[0]
        total_instances = df.shape[0]  # subset1_rows+subset2_rows

        subset1_probability = subset1_rows / total_instances
        subset2_probability = subset2_rows / total_instances

        if algorithm in ["ID3", "C4.5"]:
            threshold_gain = (
                entropy
                - subset1_probability * Training.calculateEntropy(subset1, config)
                - subset2_probability * Training.calculateEntropy(subset2, config)
            )
            subset_gains.append(threshold_gain)

        # C4.5 also need gain in the block above.
        # That's why, instead of else if we used direct if condition here
        if algorithm == "C4.5":
            threshold_splitinfo = -subset1_probability * math.log(
                subset1_probability, 2
            ) - subset2_probability * math.log(subset2_probability, 2)
            gainratio = threshold_gain / threshold_splitinfo
            subset_gainratios.append(gainratio)

        elif algorithm == "CART":
            decision_for_subset1 = subset1["Decision"].value_counts().tolist()
            decision_for_subset2 = subset2["Decision"].value_counts().tolist()

            gini_subset1 = 1
            gini_subset2 = 1

            for current_decision_for_subset1 in decision_for_subset1:
                gini_subset1 = gini_subset1 - math.pow(
                    (current_decision_for_subset1 / subset1_rows), 2
                )

            for current_decision_for_subset2 in decision_for_subset2:
                gini_subset2 = gini_subset2 - math.pow(
                    (current_decision_for_subset2 / subset2_rows), 2
                )

            gini = (subset1_rows / total_instances) * gini_subset1 + (
                subset2_rows / total_instances
            ) * gini_subset2

            subset_ginis.append(gini)

        elif algorithm == "CHAID":
            # subset1 = high, subset2 = normal

            unique_decisions = df["Decision"].unique()  # Yes, No
            num_of_decisions = len(unique_decisions)  # 2

            subset1_expected = subset1.shape[0] / num_of_decisions
            subset2_expected = subset2.shape[0] / num_of_decisions

            chi_square = 0
            for d in unique_decisions:  # Yes, No
                # decision = Yes
                subset1_d = subset1[subset1["Decision"] == d]  # high, yes
                subset2_d = subset2[subset2["Decision"] == d]  # normal, yes

                subset1_d_chi_square = math.sqrt(
                    (
                        (subset1_d.shape[0] - subset1_expected)
                        * (subset1_d.shape[0] - subset1_expected)
                    )
                    / subset1_expected
                )

                subset2_d_chi_square = math.sqrt(
                    (
                        (subset2_d.shape[0] - subset2_expected)
                        * (subset2_d.shape[0] - subset2_expected)
                    )
                    / subset2_expected
                )

                chi_square = chi_square + subset1_d_chi_square + subset2_d_chi_square

            subset_chi_squares.append(chi_square)

        # ----------------------------------
        elif algorithm == "Regression":
            superset_stdev = df["Decision"].std(ddof=0)
            subset1_stdev = subset1["Decision"].std(ddof=0)
            subset2_stdev = subset2["Decision"].std(ddof=0)

            threshold_weighted_stdev = (subset1_rows / total_instances) * subset1_stdev + (
                subset2_rows / total_instances
            ) * subset2_stdev
            threshold_reducted_stdev = superset_stdev - threshold_weighted_stdev
            subset_red_stdevs.append(threshold_reducted_stdev)

        # ----------------------------------

    if algorithm == "C4.5":
        try:
            winner_one = subset_gainratios.index(max(subset_gainratios))
        except Exception as e:
            print('ad')
    elif (
        algorithm == "ID3"
    ):  # actually, ID3 does not support for continuous features but we can still do it
        winner_one = subset_gains.index(max(subset_gains))
    elif algorithm == "CART":
        winner_one = subset_ginis.index(min(subset_ginis))
    elif algorithm == "CHAID":
        winner_one = subset_chi_squares.index(max(subset_chi_squares))
    elif algorithm == "Regression":
        winner_one = subset_red_stdevs.index(max(subset_red_stdevs))

    winner_threshold = unique_values[winner_one]
    logger.debug(f"{column_name}: {winner_threshold} in {unique_values}")

    logger.debug(f"theshold is {winner_threshold} for {column_name}")
    df[column_name] = np.where(
        df[column_name] <= winner_threshold,
        "<=" + str(winner_threshold),
        ">" + str(winner_threshold),
    )

    return df

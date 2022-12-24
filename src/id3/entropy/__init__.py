"""
This module handles calculating entropy for dataframe features.
The id3 algorithm uses min_e_feature function to get the feature with the least entropy.
"""

import numpy as np
import pandas as pd


def _eqn(val: float) -> float:
    """Calculates the entropy equation.

    Args:
        val (float): input value

    Returns:
        float: x * log(x)
    """
    return val * np.log2(val)


def _e(feat_val_df: pd.DataFrame, label: str) -> float:
    """Calculates the entropy for a feature value.

    Args:
        feat_val_df (pd.DataFrame): The
        label (str): The dataframe label column name.

    Returns:
        float: The entropy value for the feature value.
    """
    total = len(feat_val_df)
    return - sum([_eqn(len(i) / total) for n, i in feat_val_df.groupby(label)])


def feature_entropy(df: pd.DataFrame, feature: str, label: str) -> float:
    """Calculates the sum of products of entropy and probability for all values of a feature.

    Args:
        df (pd.DataFrame): Dataframe to calculate entropy for its feature.
        feature (str): The dataframe feature to calculate its entropy.
        label (str): The dataframe label column name.
    Returns:
        float: Feature entropy value.
    """
    return sum([p(df, feature, str(n)) * _e(feat_val_df, label) for n, feat_val_df in df.groupby(feature)])


def p(df: pd.DataFrame, feature: str, feature_value: str) -> float:
    """Calculates the probability of occurrence of a feature value.

    Args:
        df (pd.DataFrame): The dataset to calculate the probability for.
        feature (str): The feature to calculate probability of one of its values.
        feature_value (str): The feature value to calculate its probability.
    Returns:
        float: The probability of the occurrence of the feature value.
    """
    return len(df.loc[df[feature] == feature_value]) / len(df)


def max_info_gain_feature(df: pd.DataFrame, label: str) -> str:
    """Gets the feature with the least entropy in a dataset with respect to a label.

    Args:
        df (pd.DataFrame): The dataframe to calculate entropy for its features.
        label (str): The dataframe label column name.

    Returns:
        str: The feature with the least entropy value
    """
    entropy_values = dict(sorted({f: feature_entropy(df, f, label) for f in df}.items(), key=lambda x: x[1]))
    del entropy_values[label]
    return min(entropy_values, key=entropy_values.get)

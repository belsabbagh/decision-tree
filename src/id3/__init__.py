"""
This module contains the functions that build the decision tree using the id3 algorithm.
"""

import pandas as pd

from src.id3.entropy import max_info_gain_feature


def build_decision_tree(df: pd.DataFrame, label: str) -> dict:
    """Builds a decision tree from a dataset.

    Args:
        df (pd.DataFrame): The dataset to build the decision tree from.
        label (str): The dataset label column name.

    Returns:
        dict:  A dictionary representing the decision tree for the given dataset.
    """
    return _id3(df, label, {})


def _id3(df: pd.DataFrame, label: str, tree: dict) -> str | dict:
    """The id3 algorithm that recursively builds the decision tree.
    The algorithm stops when all the features are visited or when the given dataset label has only one unique value.
    How it works:
    1. If all the features are visited or the label has only one unique value, return the tree.
    2. Get the feature with the least entropy*proportion.
    3. For each value of the feature, create a subtree.
    4. For each subtree, call the id3 algorithm with the subset of the dataset that has the value of the feature.
    5. Return the tree.

    Args:
        df (pd.DataFrame): The dataset to build the decision tree from.
        label (str): The dataset label column name.
        tree (dict): The decision tree.

    Returns:
        str | dict: A dictionary representing the decision tree for the given dataset.
    """
    f = max_info_gain_feature(df, label)
    tree[f] = {}
    subtree = tree[f]
    if _label_has_one_unique_value(df, label):
        return _get_first_label_value(df, label)
    tree[f] = {v: _id3(_exclude_df_feature(v_grp, f), label, subtree.get(v, {})) for v, v_grp in df.groupby(f)}
    return tree


def _get_first_label_value(df: pd.DataFrame, label: str) -> str:
    """Gets the first label value.

    Args:
        df (pd.DataFrame): The dataset to get the first label value from.
        label (str): The dataset label column name.

    Returns:
        str: The first label value.
    """
    return df[label].unique()[0]


def _label_has_one_unique_value(df: pd.DataFrame, label: str) -> bool:
    """Checks if the label has only one unique value.

    Args:
        df (pd.DataFrame): The dataset to check if the label has only one unique value.
        label (str): The dataset label column name.

    Returns:
        bool: True if the label has only one unique value, False otherwise.
    """
    return len(df[label].unique()) == 1


def _exclude_df_feature(df: pd.DataFrame, f: str) -> pd.DataFrame:
    return df.loc[:, df.columns != f]

"""
This module contains the functions that build the decision tree using the id3 algorithm.
"""

import pandas as pd

from src.decision_tree.id3.entropy import max_info_gain_feature
from src.decision_tree.util import exclude_df_col


def build_decision_tree(x: pd.DataFrame, y: pd.DataFrame) -> dict:
    """Builds a decision tree from a dataset.

    Args:
        x (pd.DataFrame): The dataset feature columns.
        y (pd.DataFrame): The dataset label column.

    Returns:
        dict:  A dictionary representing the decision tree for the given dataset.
    """
    return _id3(x, y, {})


def _id3(x: pd.DataFrame, y: pd.DataFrame, tree: dict) -> str | dict:
    """The id3 algorithm that recursively builds the decision tree.
    The algorithm stops when all the features are visited or when the given dataset label has only one unique value.
    How it works:
    1. If all the features are visited or the label has only one unique value, return the tree.
    2. Get the feature with the least entropy*proportion.
    3. For each value of the feature, create a subtree.
    4. For each subtree, call the id3 algorithm with the subset of the dataset that has the value of the feature.
    5. Return the tree.

    Args:
        x (pd.DataFrame): The dataset feature columns.
        y (pd.DataFrame): The dataset label column.
        tree (dict): The decision tree.

    Returns:
        str | dict: A dictionary representing the decision tree for the given dataset.
    """
    if _label_has_one_unique_value(y):
        return _get_first_label_value(y)
    f = max_info_gain_feature(x, y)
    tree[f] = {}
    for v, v_grp in x.groupby(f):
        indexes = list(v_grp.index.values)
        tree[f][v] = _id3(
            exclude_df_col(v_grp, f),
            y[y.index.isin(indexes)],
            tree[f].get(v, {})
        )
    return tree


def _get_first_label_value(df: pd.DataFrame) -> str:
    """Gets the first label value.

    Args:
        df (pd.DataFrame): The dataset to get the first label value from.
        label (str): The dataset label column name.

    Returns:
        str: The first label value.
    """
    return df.unique()[0]


def _label_has_one_unique_value(df: pd.DataFrame) -> bool:
    """Checks if the label has only one unique value.

    Args:
        df (pd.DataFrame): The dataset to check if the label has only one unique value.
        label (str): The dataset label column name.

    Returns:
        bool: True if the label has only one unique value, False otherwise.
    """
    return len(df.unique()) == 1

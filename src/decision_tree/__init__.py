"""
This module contains the DecisionTree class.
"""

import json
import pandas as pd
from src.decision_tree.id3 import build_decision_tree
from src.decision_tree.util import exclude_df_col


class DecisionTree:
    """
    The DecisionTree class implemented similar to sklearn models.
    """
    __tree: dict | None = None

    def __init__(self):
        self.__tree = None

    def __repr__(self):
        return f'{{tree: {self.__tree}}}'

    def fit(self, x, y):
        self.__tree = build_decision_tree(x, y)
        return self

    def load_tree_json(self, path):
        with open(path, "r") as infile:
            self.__tree = json.loads(infile.read())

    def save_json(self, path):
        with open(path, "w") as outfile:
            outfile.write(self.__tree_to_json())

    def __tree_to_json(self):
        return json.dumps(self.__tree, indent=4)

    def __predict(self, x):
        return self.__traverse(x, self.__tree)

    def predict(self, x):
        return self.__predict(x)

    @ staticmethod
    def __subtree_factory(df: pd.DataFrame, node: str, value: str, subtree):
        return DecisionTree.__traverse(
            exclude_df_col(df.loc[df[node] == value], node),
            subtree
        ) if type(subtree) is dict else subtree

    @ staticmethod
    def __traverse(df: pd.DataFrame, tree: dict) -> str | int | bool | None:
        node = list(tree.keys())[0]
        for value, subtree in tree[node].items():
            return DecisionTree.__subtree_factory(df, node, value, subtree)
        return None

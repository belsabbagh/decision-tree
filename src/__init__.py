import pandas as pd
import numpy as np
import io

class Node:
    value: str
    children: dict

    def __init__(self, value: str):
        self.value = value
        self.children = {}

class Tree:
    root: Node

    def __init__(self, root: Node):
        self.root = root


def _eqn(val: float) -> float:
    return val * np.log2(val)


def _e(feat_val_df: pd.DataFrame, label: str) -> float:
    return _feature_value_entropy([len(i) for _, i in feat_val_df.groupby(label)])


def _feature_value_entropy(values_per_label_value: list[int]) -> float:
    total = sum(values_per_label_value)
    return -sum([_eqn(i / total) for i in values_per_label_value])


def p(df: pd.DataFrame, feature: str, feature_value: str) -> float:
    return len(df.loc[df[feature] == feature_value]) / len(df)


def feature_entropy(df: pd.DataFrame, f: str, label: str) -> float:
    return sum(
        [
            p(df, f, str(n)) * _e(fv_df, label)
            for n, fv_df in df.groupby(f)
        ]
    )


def max_info_gain_feature(df: pd.DataFrame, y: pd.Series) -> str:
    label = y.name
    _df = df.copy()
    _df.insert(0, label, y)
    entropy_values = {f: feature_entropy(_df, f, label) for f in _df}
    del entropy_values[label]
    return min(entropy_values, key=entropy_values.get)


def make_tree(
    df: pd.DataFrame,
    y: pd.Series,
    visited_features=set(),
    feature_picker=max_info_gain_feature,
) -> Node:
    if len(y.unique()) == 1:
        return Node(y.unique()[0])
    if len(visited_features) == len(df.columns):
        return Node(y.value_counts().idxmax())
    root = Node(max_info_gain_feature(df, y))
    root.children.update({n: Node(n) for n in df[root.value].unique()})
    for value in root.children:
        child = root.children[value]
        if child.value in visited_features:
            continue
        df_child = df.loc[df[root.value] == child.value]
        indexes = list(df_child.index.values)
        y_child = y.loc[indexes]
        root.children[value] = make_tree(
            df_child, y_child, visited_features | {root.value}, feature_picker
        )

    return root

def print_tree(tree: Tree):
    output = io.StringIO()
    def _print_tree(node: Node, prefix=""):
        print(prefix + str(node.value), file=output)
        if node.children:
            for key, child_node in node.children.items():
                print(prefix + "  |-- " + str(key), file=output)
                _print_tree(child_node, prefix + "     ")

    _print_tree(tree.root)
    res = output.getvalue()
    output.close()
    return res

class DecisionTree:
    tree: Tree

    def __init__(self, feature_picker=max_info_gain_feature):
        self.feature_picker = feature_picker

    def fit(self, df: pd.DataFrame, y: pd.Series):
        self.tree = Tree(make_tree(df, y, set(), self.feature_picker))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._predict(row) for _, row in df.iterrows())

    def _predict(self, row: pd.Series) -> str:
        node = self.tree.root
        while node.children:
            node = node.children[row[node.value]]
        return node.value

    def pprint(self):
        return print_tree(self.tree)


if __name__ == "__main__":
    df = pd.read_csv("data.csv", index_col=0)
    y = df.pop("Play_Tennis")
    model = DecisionTree(max_info_gain_feature)
    model.fit(df, y)
    model.pprint()

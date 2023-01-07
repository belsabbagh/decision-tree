import json
from src.decision_tree.id3 import build_decision_tree
from src.decision_tree.util import exclude_df_col


class DecisionTree:
    def __init__(self):
        self.__tree = None

    def __repr__(self):
        return f'{{tree: {self.__tree}}}'

    def fit(self, x, y):
        self.__tree = build_decision_tree(x, y)
        return self

    def load(self, path):
        with open(path, "r") as infile:
            self.__tree = json.loads(infile.read())

    def save(self, path):
        with open(path, "w") as outfile:
            outfile.write(self.__tree_to_json())

    def __tree_to_json(self):
        return json.dumps(self.__tree, indent=4)

    def predict(self, x):
        return self.__traverse(x, self.__tree)

    @ staticmethod
    def __traverse(df, tree: dict, node: str = None):
        if node is None:
            node = list(tree.keys())[0]
        values = tree[node]
        for value, subtree in values.items():
            if type(subtree) is dict:
                return DecisionTree.__traverse(exclude_df_col(df.loc[df[node] == value], node), subtree)
            return subtree

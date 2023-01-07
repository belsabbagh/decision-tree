import pandas as pd

from src.decision_tree.id3 import build_decision_tree
from src.decision_tree import DecisionTree


if __name__ == '__main__':
    df = pd.read_csv('data/bear.csv')
    df['index'] = df.index
    labels = df['Class']
    df = df.drop('Class', axis=1)
    tree = DecisionTree().fit(df, labels)
    print(tree)
    tree.save('out/id3-tree.json')
    res = tree.predict(df.head(1))
    print(res)

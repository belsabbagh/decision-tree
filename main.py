import pandas as pd

from src.id3 import build_decision_tree

if __name__ == '__main__':
    df = pd.read_csv('data/tennis.csv')
    tree = build_decision_tree(df, 'Play')
    print(tree)

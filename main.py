import pandas as pd

from src.id3 import id3

if __name__ == '__main__':
    df = pd.read_csv('data/tennis.csv')
    tree = id3(df, 'Play')
    print(tree)

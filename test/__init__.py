"""
The testing module.
"""
import pandas as pd

from src.decision_tree import DecisionTree


def base_test(csv_path: str, label_name: str, json_path: str) -> None:
    """_summary_

    Args:
        csv_path (str): The path to the csv file.
        label_name (str): The name of the label column.
        json_path (str): The json path to save the tree to.
    """
    df = pd.read_csv(csv_path)
    df['index'] = df.index
    labels = df[label_name]
    df = df.drop(label_name, axis=1)
    tree = DecisionTree().fit(df, labels)
    print(tree)
    tree.save_json(json_path)
    return tree

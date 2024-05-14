"""
The testing module.
"""
import pandas as pd

from src import DecisionTree


def base_test(csv_path: str, label_name: str) -> None:
    """_summary_

    Args:
        csv_path (str): The path to the csv file.
        label_name (str): The name of the label column.
        json_path (str): The path to save the tree to.
    """
    df = pd.read_csv(csv_path)
    labels = df[label_name]
    tree = DecisionTree()
    tree.fit( df.drop(label_name, axis=1, inplace=False), labels)
    return tree

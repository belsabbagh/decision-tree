from src.decision_tree import DecisionTree
from test import base_test


def test_bear():
    tree = base_test('data/bear.csv', 'Class', 'out/bear-tree.json')
    ref_tree = DecisionTree().load_tree_json(r'data\correct_trees\bear-tree.json')
    assert tree == ref_tree


def test_tennis():
    tree = base_test('data/tennis.csv', 'Play', 'out/tennis-tree.json')
    ref_tree = DecisionTree().load_tree_json(
        r'data\correct_trees\tennis-tree.json')
    assert tree == ref_tree


def test_computer():
    tree = base_test('data/computer.csv', 'Buy', 'out/computer-tree.json')
    ref_tree = DecisionTree().load_tree_json(
        r'data\correct_trees\computer-tree.json')
    assert tree == ref_tree

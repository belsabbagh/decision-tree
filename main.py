"""
The main module.
"""

from test import base_test


if __name__ == '__main__':
    base_test('data/bear.csv', 'Class', 'out/bear-tree.json')
    base_test('data/tennis.csv', 'Play', 'out/tennis-tree.json')
    base_test('data/computer.csv', 'Buy', 'out/computer-tree.json')

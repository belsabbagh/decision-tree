from src.decision_tree import DecisionTree
from test import base_test


def test_bear():
    tree = base_test('data/bear.csv', 'Class', 'out/bear-tree.json')
    assert tree.tree_to_dict() == {
        "Size": {
            "Big": "Dangerous",
            "Middle": {
                "Color": {
                    "Black": "No",
                    "Brown": {
                        "Hair": {
                            "Curls": "No",
                            "Straight": "Dangerous"
                        }
                    }
                }
            },
            "Small": {
                "Color": {
                    "Black": "No",
                    "Brown": {
                        "Hair": {
                            "Curls": "Dangerous",
                            "Straight": "No"
                        }
                    }
                }
            }
        }
    }


def test_tennis():
    tree = base_test('data/tennis.csv', 'Play', 'out/tennis-tree.json')
    assert tree.tree_to_dict() == {
        "Outlook": {
            "Overcast": "Yes",
            "Rainy": {
                "Windy": {
                    "Strong": "No",
                    "Weak": "Yes"
                }
            },
            "Sunny": {
                "Humidity": {
                    "High": "No",
                    "Normal": "Yes"
                }
            }
        }
    }


def test_computer():
    tree = base_test('data/computer.csv', 'Buy', 'out/computer-tree.json')
    assert tree.tree_to_dict() == {
        "Age": {
            "Middle": "Yes",
            "Old": {
                "Credit": {
                    "Good": "No",
                    "Ok": "Yes"
                }
            },
            "Young": {
                "Student": {
                    "No": "No",
                    "Yes": "Yes"
                }
            }
        }
    }

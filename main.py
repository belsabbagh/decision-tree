"""
The main module.
"""
import os
from test import base_test


test_cases = [
    ('data/bear.csv', 'Class'),
    ('data/tennis.csv', 'Play'),
    ('data/computer.csv', 'Buy'),
]


if __name__ == '__main__':
    for csv_path, label_name in test_cases:
        model = base_test(csv_path, label_name)
        with open(os.path.join('out', f'{os.path.basename(csv_path)[:-4]}.txt'), 'w') as f:
            f.write(model.pprint())


import numpy as np


def knn():
    _group = np.array([
        [1.0, 2.0],
        [1.2, 0.1],
        [0.1, 1.4],
        [0.3, 3.5],
        [1.1, 1.0],
        [0.5, 1.5]
    ])
    _labels = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    return _group, _labels


def unary_linear_regression():
    return np.array([
        [1, 2, 4, 6, 8],
        [2, 5, 7, 8, 9]
    ])

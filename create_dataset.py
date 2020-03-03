# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def create_dataset():
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


def test():
    group, labels = create_dataset()
    plt.scatter(group[labels == 'A', 0], group[labels == 'B', 1], color='r', marker='*')
    plt.scatter(group[labels == 'B', 0], group[labels == 'B', 1], color='g', marker='+')
    plt.show()

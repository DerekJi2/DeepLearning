import numpy as np


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

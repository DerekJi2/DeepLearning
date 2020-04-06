# -*- coding: utf-8 -*-
import numpy as np


def sampleSeeds():
    return np.array([
        [1, 2, 4, 6, 8],
        [2, 5, 7, 8, 9]
    ])


class UnaryLinearRegression(object):
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, x_train, y_train):
        numerator = 0.0
        denominator = 0.0
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)
            denominator += (x_i - x_mean) ** 2
        self.a = numerator / denominator
        self.b = y_mean - self.a * x_mean
        return self

    def predict(self, x_test_group):
        return np.array([self._predict(x_test) for x_test in x_test_group])

    def _predict(self, x_test):
        return self.a * x_test + self.b
        return self

    def mean_squared_error(self, y_true, y_predict):
        return np.sum((y_true - y_predict) ** 2) / len(y_true)

    def r_squared(self, y_true, y_predict):
        return 1 - (self.mean_squared_error(y_true, y_predict)) / np.var(y_true)

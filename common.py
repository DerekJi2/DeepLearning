import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def show_sigmoid():
    plot_x = np.linspace(-10, 10, 100)
    plot_y = sigmoid(plot_x)
    plt.plot(plot_x, plot_y)
    plt.show()
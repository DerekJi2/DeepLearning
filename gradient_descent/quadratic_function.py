import numpy as np
import matplotlib.pyplot as plt


# Loss Function
def J(theta):
    return (theta - 2.5) ** 2 - 1


# Derivative of Loss Function
def derivJ(theta):
    return 2 * (theta - 2.5)


def demo_start():
    plot_x = np.linspace(-1, 6, 141)
    plot_y = J(plot_x)
    plt.scatter(plot_x[5], plot_y[5], color='r')
    plt.plot(plot_x, plot_y)
    plt.xlabel('η', fontsize=15)
    plt.ylabel('Losses', fontsize=15)
    plt.show()

    theta_history = move_down(1.1)
    his_plot_x = np.array(theta_history)
    his_plot_y = J(his_plot_x)
    plt.plot(plot_x, plot_y, color='r')
    plt.plot(his_plot_x, his_plot_y, marker='x')
    plt.show()

    print(len(theta_history))


def move_down(eta):
    theta = 0.0
    theta_history = [theta]
    epsilon = 1e-8
    print(epsilon)
    while True:
        gradient = derivJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
    return theta_history

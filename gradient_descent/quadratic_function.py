import numpy as np
import matplotlib.pyplot as plt


# Loss Function
def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')


# Derivative of Loss Function
def derivJ(theta):
    return 2 * (theta - 2.5)


def demo_start():
    eta = 0.1
    n_iterations = 30

    plot_x = np.linspace(-1, 6, 141)
    plot_y = J(plot_x)
    plt.scatter(plot_x[5], plot_y[5], color='r')
    plt.plot(plot_x, plot_y)
    plt.xlabel('η', fontsize=15)
    plt.ylabel('Losses', fontsize=15)
    plt.show()

    theta_history = move_down(eta, n_iterations)

    his_plot_x = np.array(theta_history)
    his_plot_y = J(his_plot_x)
    plt.plot(plot_x, plot_y, color='r')
    plt.plot(his_plot_x, his_plot_y, marker='x')
    plt.show()

    print(len(theta_history))


def move_down(eta, n_iterations):
    theta = 0.0
    theta_history = [theta]
    epsilon = 1e-8
    print(epsilon)
    i_iteration = 0
    while i_iteration < n_iterations:
        gradient = derivJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        i_iteration = i_iteration + 1
        theta_history.append(theta)
        if abs(J(theta) - J(last_theta)) < epsilon:
            break
    return theta_history

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import knn_classify as knn


def load_data():
    batch_size = 100
    # MNIST dataset
    train_dataset = dsets.MNIST(root='/ml/pymnist',  # specify the root dir of the data
                                train=True,  # use TRAIN dataset
                                transform=None,  # no pro-processing
                                download=True)  # download from internet
    test_dataset = dsets.MNIST(root='/ml/pymnist',  # specify the root dir of the data
                               train=False,  # use TEST dataset
                               transform=None,  # no pro-processing
                               download=True)  # download from internet
    # Loading data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)  # shuffle data
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    return train_dataset, train_loader, test_dataset, test_loader


def print_data_details(train_dataset, test_dataset):
    print("train_data", train_dataset.data.size())
    print("train_labels", train_dataset.targets.size())

    print("test_data", test_dataset.data.size())
    print("test_labels", test_dataset.targets.size())


def draw_image(dataset, index=0):
    digit = dataset.data[index]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    print(dataset.targets[index])
    return


def knn_on_mnist(train_loader, test_loader):
    x_train = train_loader.dataset.data.numpy()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    y_train = train_loader.dataset.targets.numpy()

    x_test = test_loader.dataset.data[:1000].numpy()
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    y_test = test_loader.dataset.targets[:1000].numpy()

    num_test = y_test.shape[0]

    y_test_pred = knn.knn_classify(5, 'M', x_train, y_train, x_test)
    num_correct = np.sum(y_test_pred == y_test)

    accuracy = float(num_correct) / num_test

    print('Got %d / %d correct => accuracy: %f', (num_correct, num_test, accuracy))
    return

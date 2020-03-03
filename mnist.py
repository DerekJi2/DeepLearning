import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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

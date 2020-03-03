import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def load_data():
    batch_size = 100
    # MNIST dataset
    train_dataset = dsets.MNIST(root='/ml/pymnist',  # 选择数据的根目录
                                train=True,  # 选择训练集
                                transform=None,  # 不考虑使用任何数据预处理
                                download=True)  # 从网络上下载图片
    test_dataset = dsets.MNIST(root='/ml/pymnist',  # 选择数据的根目录
                               train=False,  # 选择测试集
                               transform=None,  # 不考虑使用任何数据预处理
                               download=True)  # 从网络上下载图片
    # 加载数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)  # 将数据打乱
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

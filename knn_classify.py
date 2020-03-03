# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import operator

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


def scatter(matrix, names):
    plt.scatter(matrix[names == 'A', 0], matrix[names == 'A', 1], color='r', marker='*')
    plt.scatter(matrix[names == 'B', 0], matrix[names == 'B', 1], color='g', marker='+')
    plt.show()
    return


def knn_classify(k, dis, X_train, x_train, Y_test):
    assert dis == 'E' or dis == 'M', 'dis must E or M，E代表欧式距离，M代表曼哈顿距离'
    num_test = Y_test.shape[0]  # 测试样本的数量
    labelers = []
    '''
   使用欧式距离公式作为距离度量
    '''
    if dis == 'E':
        for i in range(num_test):
            # 实现欧式距离公式
            tile = np.tile(Y_test[i], (X_train.shape[0], 1))
            delta = X_train - tile
            mi = delta ** 2
            _sum = np.sum(mi, axis=1)
            distances = np.sqrt(_sum)
            nearest_k = np.argsort(distances)  # 距离由小到大进行排序，并返回index值
            topK = nearest_k[:k]
            # 选取前k个距离
            classCount = {}
            for i in topK:  # 统计每个类别的个数
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labelers.append(sortedClassCount[0][0])
        return np.array(labelers)
    # 使用曼哈顿公式作为距离度量
    else:
        for i in range(num_test):
            # 实现曼哈顿距离公式
            tile = np.tile(Y_test[i], (X_train.shape[0], 1))
            delta = X_train - tile
            mi = np.abs(delta)
            distances = np.sum(mi, axis=1)
            nearest_k = np.argsort(distances)  # 距离由小到大进行排序，并返回index值
            topK = nearest_k[:k]
            # 选取前k个距离
            classCount = {}
            for i in topK:  # 统计每个类别的个数
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labelers.append(sortedClassCount[0][0])
        return np.array(labelers)


# 读者自行补充完成

def test(distance_type):
  print('\r\nDistance Type: ' + distance_type)
  group, labels = create_dataset()
  scatter(group, labels)

  new_points = np.array([[1.0, 2.1], [0.4, 2.0]]);
  y_test_pred = knn_classify(3, distance_type, group, labels, new_points)
  new_group = np.concatenate((group, new_points))
  new_labels = np.concatenate((labels, y_test_pred))
  scatter(new_group, new_labels)
  print(y_test_pred)  # 打印输出['A' 'B']，和我们的判断是相同的
  return
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
    assert dis == 'E' or dis == 'M', 'dis must be E or M'
    num_test = Y_test.shape[0]  # The number of tests
    labelers = []
    '''
       E: Euler Distance
    '''
    if dis == 'E':
        for i in range(num_test):
            # euler distance
            tile = np.tile(Y_test[i], (X_train.shape[0], 1))
            delta = X_train - tile
            mi = delta ** 2
            _sum = np.sum(mi, axis=1)
            distances = np.sqrt(_sum)
            nearest_k = np.argsort(distances)  # ORDER ASC, returns indexes
            topK = nearest_k[:k]
            # Fetch TOP K
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labelers.append(sortedClassCount[0][0])
        return np.array(labelers)
    # M: Manhattan Distance
    else:
        for i in range(num_test):
            print('knn_classify(): calculating ', i+1)
            # Manhattan Distance
            tile = np.tile(Y_test[i], (X_train.shape[0], 1))
            delta = X_train - tile
            mi = np.abs(delta)
            distances = np.sum(mi, axis=1)
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            # Fetch TOP K
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            labelers.append(sortedClassCount[0][0])
        return np.array(labelers)


def getXmean(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    mean_image = np.mean(x_train, axis=0)
    return mean_image


def centralized(x_test, mean_image):
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_test = x_test.astype(np.float)
    x_test -= mean_image
    return x_test


def test(distance_type):
  print('\r\nDistance Type: ' + distance_type)
  group, labels = create_dataset()
  scatter(group, labels)

  new_points = np.array([[1.0, 2.1], [0.4, 2.0]]);
  y_test_pred = knn_classify(3, distance_type, group, labels, new_points)
  new_group = np.concatenate((group, new_points))
  new_labels = np.concatenate((labels, y_test_pred))
  scatter(new_group, new_labels)
  print(y_test_pred)  # Expecting ['A' 'B']
  return

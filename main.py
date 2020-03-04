# -*- coding: utf-8 -*-

import create_dataset as cd
import knn_classify as knn
import mnist

# knn.test('E')
# knn.test('M')

train_dataset, train_loader, test_dataset, test_loader = mnist.load_data()

# mnist.print_data_details(train_dataset, test_dataset)

mnist.draw_image(train_loader.dataset, 989)

# mnist.knn_on_mnist(train_loader, test_loader)

# mnist.centralized_knn_on_mnist(train_loader, test_loader)

mnist.draw_centralized_image(train_dataset, test_dataset, 989)

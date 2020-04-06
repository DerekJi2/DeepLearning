# -*- coding: utf-8 -*-
import common
import knn
import mnist
import unary_linear_regression as reg

# knn.test('E')
# knn.test('M')

# train_dataset, train_loader, test_dataset, test_loader = mnist.load_data()
# mnist.print_data_details(train_dataset, test_dataset)
# mnist.draw_image(train_loader.dataset, 989)
# mnist.knn_on_mnist(train_loader, test_loader)
# mnist.centralized_knn_on_mnist(train_loader, test_loader)
# mnist.draw_centralized_image(train_dataset, test_dataset, 989)

# reg.test()

common.show_sigmoid()

print('All done!')

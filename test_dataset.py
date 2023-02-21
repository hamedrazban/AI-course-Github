import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# # test Iris flower dataset
# data = load_iris()
# print('Iris data input shape',data.data.shape)
# print('Iris data target shape',data.target.shape)
# print('Iris target[1,50,100] =', data.target[[1, 50, 100]])
#
# # test Diabetes dataset
# data = load_diabetes()
# print('\nDiabetes data input shape',data.data.shape)
# print('Diabetes data target shape',data.target.shape)
# print('Diabetes target max:', max(data.target))
# print('Diabetes target min:', min(data.target))
#
#
# # test make_classification sample generator
# X, Y = make_classification(n_samples=400, n_features=2, n_redundant=0,
#                            n_informative=2, n_clusters_per_class=1,
#                            class_sep=2, n_classes=4)
# my_scatter =plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=30, edgecolor="k")
# print('\nmake_classification X.shape', X.shape)
# print('make_classification Y.shape', Y.shape)
# plt.show()
#
#
# # test make_regression sample generator
# X, y = make_regression(
#     n_samples=200,
#     n_features=1,
#     n_informative=1,
#     n_targets=1,
#     bias=2,
#     noise=10,
# )
# my_scatter =plt.scatter(X, y, marker="o", c='g', s=30, edgecolor="k")
# plt.show()
#
#
# #test train_test_split function
# n_samples = 300
# rng = np.random.RandomState(0)
# X = rng.randn(n_samples, 1)
# noise = rng.normal(loc=0.0, scale=0.1, size=n_samples)
# y = X + np.sin( np.pi * X) - noise.reshape(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# print('\ntrain_test_split X_train shape ', X_train.shape)
# print('train_test_split X_test shape ', X_test.shape)
# print('train_test_split Y_train shape ', y_train.shape)
# print('train_test_split Y_test shape ', y_test.shape)
#
# plt.scatter(X_train, y_train, marker="o", c='r', s=30, edgecolor="k", label='Train data')
# plt.scatter(X_test, y_test, marker="o", c='g', s=30, edgecolor="k", label='Test data')
# plt.legend()
# plt.show()


# #test StandardScaler
# data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
# scaler = StandardScaler().fit(data)
# print('scaler.mean: ',scaler.mean_)
# print('scaler.var: ',scaler.var_)
# print('data transformed: ', scaler.transform(data))

#test minmax_scale
data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
scaler = MinMaxScaler().fit(data)
print('scaler.data_max_: ',scaler.data_max_)
print('scaler.data_min_: ',scaler.data_min_)
print('data transformed: ', scaler.transform(data))

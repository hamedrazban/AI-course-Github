import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



# test MLPClassifier
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 1, 1, 0]
clf = MLPClassifier(hidden_layer_sizes=(3, 2),
                    solver='sgd',
                    activation='logistic',
                    batch_size=4,
                    learning_rate='constant',
                    learning_rate_init=0.05,
                    max_iter=200,
                    shuffle= True)

params = {'activation':'tanh','max_iter':1000}
clf.set_params(**params)
clf.fit(X,y)

print('clf.classes_: ', clf.classes_)
print('clf.coefs_: ', clf.coefs_)
print('clf.intercepts_: ', clf.intercepts_)
print('get_params:', clf.get_params())

print('clf(', X, ')=')
print(clf.predict(X))
print('score: ', clf.score(X, y))

plt.plot(clf.loss_curve_,'r*-')
plt.show()

#A multi-class classification example
X, Y = make_classification(n_samples=800,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           n_clusters_per_class=1,
                           class_sep=1.2,
                           n_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test1 = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(6),
                    solver='sgd',
                    activation='tanh',
                    batch_size=200,
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    max_iter=800,
                    shuffle= True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test1)
print('score: ', clf.score(X_test1, y_test))
plt.scatter(X_test[:,0], X_test[:,1], marker="o", c=y_pred, s=60, edgecolor='k')
plt.scatter(X_test[:,0], X_test[:,1], marker="o", c=y_test, s=10, edgecolor=None)
plt.show()


# A regression example
# n_samples = 800
# rng = np.random.RandomState(0)
# X = rng.randn(n_samples, 1)
# noise = rng.normal(loc=0.0, scale=0.03, size=n_samples)
# y = (X + np.sin( np.pi * X) - noise.reshape(X.shape)).ravel()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test1 = scaler.transform(X_test)
#
# rgr = MLPRegressor(hidden_layer_sizes=(6),
#                     solver='adam',
#                     activation='tanh',
#                     batch_size=200,
#                     learning_rate='constant',
#                     learning_rate_init=0.05,
#                     max_iter=300,
#                     shuffle= True)
#
# rgr.fit(X_train,y_train)
# y_pred = rgr.predict(X_test1)
# print('score: ', rgr.score(X_test1, y_test))
#
# print('MAE: ', mean_absolute_error(y_test, y_pred))
# print('MSE: ', mean_squared_error(y_test, y_pred))
# print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
#
# plt.scatter(X_test, y_pred, marker="o", c='r', s=40, edgecolor="k", label='y_pred')
# plt.scatter(X_test, y_test, marker="o", c='g', s=30, edgecolor="k", label='y_test')
# plt.legend()
# plt.show()

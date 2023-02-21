import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import joblib



# A regression example
n_samples = 800
rng = np.random.RandomState(0)
X = rng.randn(n_samples, 1)
noise = rng.normal(loc=0.0, scale=0.03, size=n_samples)
y = (X + np.sin( np.pi * X) - noise.reshape(X.shape)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test1 = scaler.transform(X_test)

rgr = MLPRegressor(hidden_layer_sizes=(6),
                    solver='adam',
                    activation='tanh',
                    batch_size=200,
                    learning_rate='constant',
                    learning_rate_init=0.05,
                    max_iter=300,
                    shuffle= True)

rgr.fit(X_train,y_train)
y_pred = rgr.predict(X_test1)
print('score: ', rgr.score(X_test1, y_test))

print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))

plt.scatter(X_test, y_pred, marker="o", c='r', s=40, edgecolor="k", label='y_pred')
plt.scatter(X_test, y_test, marker="o", c='g', s=30, edgecolor="k", label='y_test')
plt.legend()
plt.show()

pickle.dump(rgr, open('my_regression_model.pcl', 'wb'))
joblib.dump(rgr,'my_regression_model.jbl')
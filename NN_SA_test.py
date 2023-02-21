import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

################################
# data generation
sample_num = 400
data_input = np.random.uniform(low=0, high=1, size=(sample_num, 3))
data_output = np.sin(data_input[:,0]) + 7*np.sin(data_input[:,1]) + 0.1*data_input[:,2]**4*np.sin(data_input[:,0])

#trainingset and testingset
X_train, X_test, y_train, y_test = train_test_split (data_input, data_output, test_size =0.2, shuffle=True)

#NN model construction
rgr1 = MLPRegressor(hidden_layer_sizes=(6),
                    solver='adam',
                    activation='tanh',
                    batch_size=40,
                    learning_rate='constant',
                    learning_rate_init=0.05,
                    max_iter=300,
                    shuffle= True)
rgr1.fit(X_train, y_train)

score1 = rgr1.score(X_test, y_test)
print('NN model with full inputs score is:', score1)
print('A test example, given X:', X_test[0,:])
print('Its true value is ', np.sin(X_test[0,0]) + 7*np.sin(X_test[0,1]) + 0.1*X_test[0,2]**4*np.sin(X_test[0,0]))
print('Its prediction value is ', rgr1.predict(X_test[0,:].reshape(1,-1)))
print('#################################')

################################ now we redo the above process, but use only x1 and x2 as NN model inputs
#trainingset and testingset
X_train, X_test, y_train, y_test = train_test_split (data_input[:,:-1], data_output, test_size =0.2, shuffle=True)

#NN model construction
rgr2 = MLPRegressor(hidden_layer_sizes=(6),
                    solver='adam',
                    activation='tanh',
                    batch_size=40,
                    learning_rate='constant',
                    learning_rate_init=0.05,
                    max_iter=300,
                    shuffle= True)
rgr2.fit(X_train, y_train)

score2 = rgr2.score(X_test, y_test)
print('NN model with x1 and x2 inputs score is:', score2)


# plot the prediction points of rgr2 in 3D space (x1, x2 and y) and compare with the simplified base surface y = sin(x1) + 7*sin(x2)
X1 = np.arange(0, 1, 0.1)
X2 = np.arange(0, 1, 0.1)
X1, X2 = np.meshgrid(X1, X2)
Z = np.sin(X1) + 7*np.sin(X2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the base plane.
plane = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Plot the regression points in 3D
row, col = X1.shape
for i in range(row):
    for j in range(col):
        pred = rgr2.predict(np.array([X1[i,j], X2[i,j]]).reshape(1,-1))
        ax.scatter(X1[i,j], X2[i,j], pred, color = "green")

plt.show()
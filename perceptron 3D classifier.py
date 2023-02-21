import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm


#https://developpaper.com/perceptron-tutorial-implementation-and-visual-examples/
class perceptron():
    def __init__(self, N=3, alpha=0.1):
        '''
        :param N: number of perceptron input
        :param alpha: learning rate
        '''
        self.N = N
        self.alpha = alpha
        self.bias = -1
        self.W = np.random.rand(N+1) # the extra one for bias weight, W[0] for bias, W[1] for x, W[2] for y, W[3] for z


    def forward(self, input):
        '''
        :param input: a Nby3 2D array, each row contains x, y, z, but not includes bias
        '''
        assert input.size==self.N, 'input size error, no need bias'

        X = np.insert(input, 0, self.bias) #bials, x, y, z
        g = np.dot(X, self.W)
        if g>=0:
            return 1
        else:
            return -1


    def train(self, data, n_iter=100): #see the algorithm on pp.30 of lecture note 2
        '''
        :param
        data: N by 3, for each row = x pos, y pos, z pos, class label
        :return: trained weights
        '''
        n_samples = data.shape[0]
        bias_vector = np.ones((n_samples, 1)) * self.bias
        print('bias_vector ', bias_vector.shape)
        X = np.hstack((bias_vector, data))

        for i in range(n_iter):
            for j in range(n_samples):
                y= self.forward(X[j,1:-1]) #predicted value
                self.W += self.alpha* X[j,:-1] * ((X[j,-1])-y) #X[j, -1] * X[j, :-1]


    def print_weight(self):
        print('Weight: ', self.W)

    def get_weight(self):
        return self.W

    def set_weight(self, w):
        self.W = w


#####################main
# 1. Dataset generation
# Gaven the based plane: x+y+z=0
sample_num = 200
training_iteration = 100
data_input = np.random.uniform(low=-10, high=10, size=(sample_num, 3)) # the sample point in 3D space
data_input[:,-1] = np.random.uniform(low=-20, high=20, size=(sample_num, 1)).reshape(-1)

data_true_class = np.zeros((sample_num,1)) # the sample's label, either 1 or -1

base_plane = data_input[:, 0] + data_input[:, 1] + data_input[:, 2] # the base plane x+y+z=0
data_true_class[base_plane>=0] = 1 # points that are above the plane
data_true_class[base_plane < 0] = -1 # points that are below the plane

data_class1 = data_input[data_true_class.reshape(-1)==1, :]  # collect the class 1's data for plot purpose
data_class2 = data_input[data_true_class.reshape(-1)==-1, :]  # collect the class 1's data for plot purpose
print(data_class2.shape)
print(data_class1.shape)

# plot the data samples in 2 colors
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
X, Y = np.meshgrid(X, Y)
Z = -X-Y

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the base plane.
plane = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Plot the two classes of sample points
ax.scatter(data_class1[:,0], data_class1[:,1], data_class1[:,2], color = "green")
ax.scatter(data_class2[:,0], data_class2[:,1], data_class2[:,2], color = "blue")

plt.show()

# 2. Perceptron model
dataset = np.hstack((data_input, data_true_class)) # form each row of the dataset as: x,y,z,label
training_num = int(sample_num*0.8)
testing_num = sample_num - training_num
trainingset = dataset[:training_num, :]
testingset = dataset[training_num:, :]
print('trainingset, ', trainingset.shape)
print('testingset', testingset.shape)

binary_classifier = perceptron(N=3, alpha=0.05)
binary_classifier.train(trainingset, n_iter=training_iteration)

# 3. Perceptron model evaluation
TP, TN, FP, FN = 0, 0, 0, 0

# use the testing set to calculate Perceptron accuracy

for i in range(testing_num):
    if binary_classifier.forward(testingset[i,:-1]) == 1 and testingset[i,-1] == 1:
        TP += 1
    if binary_classifier.forward(testingset[i, :-1]) == 1 and testingset[i, -1] == -1:
        FP += 1
    if binary_classifier.forward(testingset[i, :-1]) == -1 and testingset[i, -1] == -1:
        TN += 1
    if binary_classifier.forward(testingset[i, :-1]) == -1 and testingset[i, -1] == 1:
        FN += 1

acc = (TP + TN)/(TP + TN + FP + FN)
print('TP=', TP)
print('TN=', TN)
print('FP=', FP)
print('FN=', FN)
print('acc=', acc)



# plot the testing result
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the base plane.
base_plane_plot = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Plot the plane trained from the Perceptron model
trained_weights = binary_classifier.get_weight()
print('the trained classification plane: ', trained_weights[0], '+',
      trained_weights[1], '*x+',
      trained_weights[2], '*y+',
      trained_weights[3], '*z=0')
X1 = np.arange(-10, 10, 1)
Y1 = np.arange(-10, 10, 1)
X1, Y1 = np.meshgrid(X, Y)
Z1 = -(trained_weights[0] + trained_weights[1]*X1 + trained_weights[2]*Y1)/trained_weights[3] #since general plane equation is w0 + w1*x + w2*y + w3*z=0
trained_plane_plot = ax2.plot_surface(X1, Y1, Z1, color='m', linewidth=0, antialiased=False)

# Plot the sample points in testset, red points mean correctly classified points, while black points mean mis-classified point,
for i in range(testing_num):
    if (binary_classifier.forward(testingset[i,:-1]) == 1 and testingset[i,-1] == 1) \
            or (binary_classifier.forward(testingset[i, :-1]) == -1 and testingset[i, -1] == -1): # TP and TN cases
        ax2.scatter(testingset[i, 0], testingset[i, 1], testingset[i, 2], color="r")
    if (binary_classifier.forward(testingset[i, :-1]) == 1 and testingset[i, -1] == -1) \
            or (binary_classifier.forward(testingset[i, :-1]) == -1 and testingset[i, -1] == 1):  # FP and FN cases
        ax2.scatter(testingset[i, 0], testingset[i, 1], testingset[i, 2], color="k")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

class perceptron():
    def __init__(self, input_num =3, alpha=0.1):
        self.bias = -1
        self.input_num = input_num
        self.w = np.random.rand(input_num+1)
        self.alpha = alpha

    def forward(self, input):#activation function
        '''
            :param input: a Nby3 2D array, each row contains x, y, z, but not includes bias
        '''
        assert input.size == self.input_num, 'input size error, no need bias'

        X = np.insert(input, 0, self.bias)  # bials, x, y, z
        g = np.dot(X, self.w)
        if g >= 0:
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
                y= self.forward(X[j,1:-1]) #predicted value-----activation function
                self.w += self.alpha* X[j,:-1] * ((X[j,-1])-y) #X[j, -1] * X[j, :-1]----loss function

    def print_weight(self):
        print('Weight: ', self.w)

    def get_weight(self):
        return self.w
####### now producing data:features (x,y,z),classes (1,-1) or bellow the plane or up the plane
sample_num = 200
data_input = np.random.uniform(low=-10, high=10, size=(sample_num, 3)) # x,y, z
data_input[:,-1] =np.random.uniform(low=-20, high=20, size=(sample_num, 1)).reshape(-1) # update z
print(data_input.shape)

base_plane = data_input[:,0] + data_input[:,1] + data_input[:,2]#x+y+z=0 is the plane
print(len(base_plane))

data_true_label = np.zeros((sample_num,1))

for i in range(len(base_plane)):
    if base_plane[i] >=0:
        data_true_label[i] = 1
    else:
        data_true_label[i] = -1

dataset = np.hstack((data_input, data_true_label))#data true lable=x[j,-1]
print(dataset.shape)

traning_num = int(sample_num*0.8)
trainingset = dataset[:traning_num, :]
testing_num = sample_num - traning_num
testingset = dataset[traning_num:, :]

#plot the base plane
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
# print('Y ', Y.shape)
X, Y = np.meshgrid(X, Y)
# print('Y ', Y.shape)
# print(Y)
# print(X)
Z = -X-Y
# print(Z)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)

for i in range(len(base_plane)):
    if data_true_label[i] == 1:
        ax.scatter(dataset[i,0],dataset[i,1], dataset[i,2] , color = 'r')
    else:
        ax.scatter(dataset[i, 0], dataset[i, 1], dataset[i, 2], color='b')
plt.show()


p = perceptron(input_num =3, alpha=0.1)

p.train(trainingset)
p.print_weight()


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(testing_num):
    predict = p.forward(testingset[i,0:-1])
    if predict == 1 and testingset[i,-1] == 1 : # TP
        TP += 1
    if predict == 1 and testingset[i, -1] == -1:  # FP
            FP += 1
    if predict == -1 and testingset[i,-1] == -1 : # TN
        TN += 1
    if predict == -1 and testingset[i,-1] == 1 : # FN
        FN += 1

acc = (TP + TN) / (TP + TN + FP + FN)
print(acc)




#plot the base plane
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
print('Y ', Y.shape)
X, Y = np.meshgrid(X, Y)
print('Y ', Y.shape)

Z = -X-Y

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, color = 'r')

trained_weights = p.get_weight()
X1 = np.arange(-10, 10, 1)
Y1 = np.arange(-10, 10, 1)
X1, Y1 = np.meshgrid(X, Y)
Z1 = -(trained_weights[0] + trained_weights[1]*X1 + trained_weights[2]*Y1)/trained_weights[3] #since general plane equation is w0 + w1*x + w2*y + w3*z=0
trained_plane_plot = ax.plot_surface(X1, Y1, Z1, color='m', linewidth=0, antialiased=False)

plt.show()

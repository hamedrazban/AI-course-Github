from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
# read the data file
iris=(pd.read_csv('iris.data',sep=',')).values
# split the input and output
x=np.zeros((149,4))
y=[]
for i in range(149):
    y.append(iris[i,4])
    for j in range(4):
        x[i,j]=iris[i,j]
y=np.array(y)
# split data set to train and test sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,shuffle=True)
scaler=StandardScaler().fit(x_train)
x_train_N=scaler.transform(x_train)
x_test_N=scaler.transform(x_test)
# print(x_train.shape)      `
n=30
scores=np.zeros(n)
scores_N=np.zeros(n)
scores_s=np.zeros(n)
perceptron=np.zeros(n)
# using for loop to test the number of perceptrons
for i in range(n):
    clf=MLPClassifier(hidden_layer_sizes=(3+i*2),solver='sgd',activation='logistic',batch_size=3,learning_rate='constant',learning_rate_init=0.001,max_iter=1000,shuffle=True)
    clf_N=MLPClassifier(hidden_layer_sizes=(3+i*2),solver='sgd',activation='logistic',batch_size=3,learning_rate='constant',learning_rate_init=0.001,max_iter=1000,shuffle=True)
    clf_s=MLPClassifier(hidden_layer_sizes=(3+i*2),solver='adam',activation='relu',batch_size=3,learning_rate='constant',learning_rate_init=0.001,max_iter=1000,shuffle=True)
    clf.fit(x_train,y_train)
    clf_N.fit(x_train_N,y_train)
    clf_s.fit(x_train_N,y_train)
    y_predict=clf.predict(x_test)
    y_predict_N=clf_N.predict(x_test_N)
    y_predict_s=clf_s.predict(x_test_N)
    score=clf.score(x_test,y_test)
    score_N=clf_N.score(x_test_N,y_test)
    score_s=clf_s.score(x_test_N,y_test)
    scores[i]=score
    scores_N[i]=score_N
    scores_s[i]=score_s
    perceptron[i]=3+2*i
fig=plt.figure(figsize=(8,8))
ax1=fig.add_subplot(211)
ax1.plot(perceptron,scores,'r--',label="unnormalized input")
ax1.plot(perceptron,scores_N,'b-',label="normalized input")
ax1.plot(perceptron,scores_s,'g--',label="adam solver")
ax1.set_title("network's score on different number of perceptrons")
ax1.set_xlabel("number of perceptrons")
ax1.set_ylabel("scores")
ax1.legend()
# plot loss curve
ax2=fig.add_subplot(212)
ax2.plot(clf.loss_curve_,'r*',label="unnormalized input")
ax2.plot(clf_N.loss_curve_,'b*',label="normalized input")
ax2.plot(clf_s.loss_curve_,'g*',label="adam solver")
ax2.set_title("loss curve")
ax2.set_xlabel("iteration number")
ax2.legend()
# cm=metrics.confusion_matrix(y_test,y_predict)
cm_N=metrics.confusion_matrix(y_test,y_predict_N)
print('report for not normaized data: ',metrics.classification_report(y_test,y_predict))
print('report for normaized data: ',metrics.classification_report(y_test,y_predict_N))
# writing reports down to a file
f=open('report.txt','a')
f.write('report for not normalized data: ')
s=metrics.classification_report(y_test,y_predict)
f.write(str(s))
f.write('\n')
f.write('report for normalized data: ')
s2=metrics.classification_report(y_test,y_predict_N)
f.write(str(s2))

# metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=('0','1',2)).plot()
metrics.ConfusionMatrixDisplay(confusion_matrix=cm_N,display_labels=('0','1',2)).plot()
plt.show()

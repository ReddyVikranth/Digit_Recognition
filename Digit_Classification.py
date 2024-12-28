# importing some important libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
#training data, 80% of the total dataset
x_train = data[:33600, 1:].T
x_train = x_train.astype('float64')
y_train = data[:33600, 0]
y_train = y_train.astype('float64')
#testing data, the remaining 20% of the dataset
x_test = data[33600: , 1:].T
x_test = x_test.astype('float64')
y_test = data[33600: , 0]
y_test = y_test.astype('float64')

#initializing of the parameters (architecture of the neural network)
def init_para():
    w1 = np.full((28, 784) , (2/812)**0.5 , dtype = 'float64')
    b1 = np.zeros((28,1),dtype='float64')
    w2 = np.full((10, 28) , (2/38)**0.5 , dtype = 'float64')
    b2 = np.zeros((28,1),dtype='float64')
    return w1, b1, w2, b2

# forward propogation
def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    maxi = np.max(z)
    z = np.exp(z - maxi)
    return z / np.sum(z)
    
def forward_prop(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1 # even though the dimensions of b1 are not right, the broadcasting is done by numpy
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2 # even though the dimensions of b2 are not right, the broadcating is done by numpy
    a2 = softmax(z2)
    return z1, a1, z2, a2
# this is all about the forward propogation. I am confident based on the knowledge I have in the dimensions of the np.array

# back propogation
def labels(y): # this will make the labels suitable for the back propogation
    output = np.zeros((y.size , int(y.max()) + 1))
    output[np.arange(y.size) , y.astype('int64')] = 1
    output = output.T
    return output

def der_ReLU(z):
    return z > 0

def back_prop(a1, a2, z1, z2, w2, x, y):
    m = y.size
    y = labels(y)
    dz2 = a2 - y
    dw2 = (1/m)*(np.dot(dz2, a1.T))
    db2 = (1/m) * np.sum(dz2,+ 1)
    dz1 = np.dot(w2.T, dz2) * der_ReLU(z1)
    dw1 = (1/m) * np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, 1)
    return dw1, db1, dw2, db2
# with this, the back propogation is just done

# updating the parameters
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, a):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2
    b2 = b2 - a * db2
    return w1,  b1, w2, b2

# making the gradient descent by combining all the above functions
def gradient_descent(x, y, iterations, a):
    w1, b1, w2, b2 = init_para()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(a1, a2, z1, z2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, a)
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(x_train , y_train, 100, 0.1)
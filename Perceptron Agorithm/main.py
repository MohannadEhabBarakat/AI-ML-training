import pandas 
import numpy as np
from matplotlib import pyplot

dt = pandas.read_csv('data.csv')

pyplot.plot(dt["x0"],dt["x1"])
# print(dt["x0"])


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


# x = data x0,x1
# y = label 0 or 1
# W = wight w0, w1 
# b = bias 
def perceptronStep(X, y, W, b, learn_rate = 0.01): 

    for i in range(len(X["x0"])):
        predect = prediction(X[i],W,b)
        direction = y[i]*predect
        if predect is not y[i]:
            W[0]+=X[i]["x0"]*learn_rate*direction
            W[1]+=X[i]["x1"]*learn_rate*direction
            b+=learn_rate*direction
    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
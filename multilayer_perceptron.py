import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
inertia = 0.2
iterations = 10000
learning_rate = 1

class multilayer_perceptron:
    def __init__(self, hidden_layers, neurons_per_layer, random_scale = 1):
        self.w = np.random.rand(hidden_layers+1, neurons_per_layer+1, neurons_per_layer+1) * random_scale
        self.w[hidden_layers] = np.zeros([neurons_per_layer+1, neurons_per_layer+1])
        self.w[hidden_layers][1] = np.random.rand(neurons_per_layer+1) * random_scale
        self.last_delta_w = np.zeros([hidden_layers+1, neurons_per_layer+1, neurons_per_layer+1])
        self.o = np.zeros([hidden_layers+1, neurons_per_layer+1])
        self.sigma = self.o.copy()
        for i in range(hidden_layers+1):
            self.o[i][0] = 1
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.random_scale = random_scale

    def __str__(self):
        s = ''
        s += 'This is a multilayer perceptron.\n'
        s += 'weights:\n'
        s += str(self.w)
        s += '\nsigma:\n'
        s += str(self.sigma)
        return s

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid2(self, v):
        try:
            return 1 / (1+math.exp(-v))
        except OverflowError:
            if v > 0:
                return 1
            else:
                return 0

    def predict(self, Xs, Ys):
        Z = np.zeros(Xs.shape)
        for i in range(Xs.shape[0]):
            for j in range(Xs.shape[1]):
                value = self.forward([Xs[i,j], Ys[i, j]])
                Z[i,j] = value
        return Z

    def extend_input_x(self, x):
        extended_x = np.zeros(self.neurons_per_layer+1)
        extended_x[0] = 1
        input_n = len(x)
        extended_x[1:input_n+1] = x
        return extended_x

    def forward(self, x, debug = False):
        extended_x = self.extend_input_x(x)
        for i in range(self.hidden_layers+1):
            for j in range(1, self.neurons_per_layer+1):
                if i == self.hidden_layers and j > 1:
                    break
                if i == 0:
                    if debug:
                        print('type of w[i,j], ', type(self.w[i, j]))
                        print('type of x,', type(x))
                        print('type of x[1],', type(x[1]))
                        print('x: ', x)
                    self.o[i, j] = self.sigmoid(self.w[i, j].dot(extended_x))
                else:
                    self.o[i, j] = self.sigmoid(self.w[i, j].dot(self.o[i-1]))
        return self.o[self.hidden_layers, 1]

    def classify(self, x, debug=False):
        c = int(self.forward(x) > 0.5)
        if debug:
            print(c)
        return c

    def test_accuracy(self, X, Y, debug=False):
        n_samples = len(Y)
        correct = 0
        for i in range(n_samples):
            if debug:
                print(X[i], int(Y[i]), self.forward(X[i]))
            if self.classify(X[i]) == int(Y[i]):
                correct += 1
        correct_rate = correct / n_samples
        print('Correct rate = {}%'.format(correct_rate * 100))
        return correct_rate

    def show_decision_boundary(self, x_min, x_max, y_min, y_max):
        X1 = np.linspace(x_min, x_max, 100)
        X2 = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(X1, X2)
        Z = self.predict(X, Y)

        plt.contour(X, Y, Z, levels=[0.5])


    def backpropagation(self, X, Y, learning_rate):
        n_samples = len(Y)
        for sample in range(n_samples):
            x = X[sample]
            y = Y[sample]
            #extended_x = np.insert(x, 0, 1, axis=0) # add 1 as first element
            extended_x = self.extend_input_x(x)
            f = self.forward(x)
            #e = 0.5*(y-f)**2
            for i in range(self.hidden_layers, -1, -1):
                for j in range(1,self.neurons_per_layer+1):
                    if i == self.hidden_layers and j > 1:
                        break
                    if i == self.hidden_layers:
                        self.sigma[i, j] = (self.o[i, j]-y)*self.o[i, j]*(1-self.o[i, j])
                    else:
                        self.sigma[i, j] = self.sigma[i+1].dot(self.w[i+1][:,j]) * self.o[i, j] * (1 - self.o[i, j])
                    for k in range(self.neurons_per_layer+1):
                        if i == 0:
                            self.last_delta_w[i, j, k] = inertia*self.last_delta_w[i, j, k] + (1-inertia)*learning_rate * extended_x[k] * self.sigma[i, j]
                            self.w[i, j, k] -= self.last_delta_w[i, j, k]
                        else:
                            self.last_delta_w[i, j, k] = inertia*self.last_delta_w[i, j, k] + (1-inertia)*learning_rate * self.o[i-1, k] * self.sigma[i, j]
                            self.w[i, j, k] -= self.last_delta_w[i, j, k]

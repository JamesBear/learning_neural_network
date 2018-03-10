

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

inertia = 0.1
step_size = 0.1
iterations = 200
TRAIN_RATIO_INVERSE = 3 # inverse of 1/3 is 3
file_name = 'moon_dataset_pairs3000_r10_w6_d-4.npy'
ds = np.load(file_name)
train_length = len(ds)//TRAIN_RATIO_INVERSE
print('train_length = ', train_length)
train_ds = ds[:train_length]
test_ds = ds[train_length:]

def show_samples(dataset):
    A = dataset[dataset[:,2] == 0]
    B = dataset[dataset[:,2] == 1]

    x0 = A[:,0]
    y0 = A[:,1]
    x1 = B[:,0]
    y1 = B[:,1]
    plt.scatter(x0, y0, marker='x')
    plt.scatter(x1, y1, marker='x')

def draw_decision_boundary(w):
    x = np.array([-13-2, 23+2])
    f = lambda x:(-(w[0]-0.5)/w[2] - x*w[1]/w[2])
    #f = lambda x:(w[0] + x*w[1])
    y = np.apply_along_axis(f, 0, x)
    plt.plot(x, y, color='r')

def score(w, dataset):
    correct = 0
    x = np.zeros(3)
    x[0] = 1
    for row in dataset:
        x[1] = row[0]
        x[2] = row[1]
        if w.T.dot(x) >= 0.5:
            y = 1
        else:
            y = 0
        if y == int(row[2]):
            correct += 1
    accuracy = correct / len(dataset)
    print('accuracy is: {}%'.format(accuracy*100))
    return accuracy


def run_least_mean_squares(dataset):
    for iteration in range(iterations):
        for i in range(N):
            e_n = y[i] - h.dot(X[i])
            #print(e_n)
            h = h + step_size * e_n*X[i]

    print('h = ', h)

    draw_decision_boundary(h)

    return h


class multilayer_perceptron:
    def __init__(self, hidden_layers, neurons_per_layer):
        self.w = np.random.rand(hidden_layers+1, neurons_per_layer, neurons_per_layer)
        self.w[hidden_layers] = np.zeros([neurons_per_layer, neurons_per_layer])
        self.w[hidden_layers][0] = np.random.rand(neurons_per_layer)
        self.last_delta_w = np.zeros([hidden_layers+1, neurons_per_layer, neurons_per_layer])
        self.o = np.zeros([hidden_layers+1, neurons_per_layer])
        self.sigma = self.o.copy()
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer

    def __str__(self):
        s = ''
        s += 'This is a multilayer perceptron.\n'
        s += 'weights:\n'
        s += str(self.w)
        return s

    def sigma(self, j):
        pass

    def sigmoid(self, v):
        try:
            return 1 / (1+math.exp(-v))
        except OverflowError:
            if v < 0:
                return 0
            else:
                return 1

    def forward(self, x):
        for i in range(self.hidden_layers+1):
            for j in range(self.neurons_per_layer):
                if i == self.hidden_layers and j > 0:
                    break
                if i == 0:
                    self.o[i, j] = self.sigmoid(self.w[i, j].dot(x))
                else:
                    self.o[i, j] = self.sigmoid(self.w[i, j].dot(self.o[i-1]))
        return self.o[self.hidden_layers, 0]

    def classify(self, x):
        return int(self.forward(x) > 0.5)

    def test_accuracy(self, X, Y):
        n_samples = len(Y)
        correct = 0
        for i in range(n_samples):
            if self.classify(X[i]) == int(Y[i]):
                correct += 1
        correct_rate = correct / n_samples
        print('Correct rate = {}%'.format(correct_rate * 100))
        return correct_rate

    def show_decision_boundary(self, x_min, x_max, y_min, y_max):
        X1 = np.linspace(x_min, x_max, 100)
        X2 = np.linspace(y_min, y_max, 100)
        b_x1 = []
        b_x2 = []

        for i in X1:
            for j in X2:
                y = self.forward([1, i, j])
                if abs(y-0.5) < 0.1:
                    b_x1.append(i)
                    b_x2.append(j)
        if len(b_x1) > 0:
            plt.scatter(b_x1, b_x2, color='g')


    def backpropagation(self, X, Y, learning_rate):
        n_samples = len(Y)
        for sample in range(n_samples):
            x = X[sample]
            y = Y[sample]
            f = self.forward(x)
            #e = 0.5*(y-f)**2
            for i in range(self.hidden_layers, -1, -1):
                for j in range(self.neurons_per_layer):
                    if i == self.hidden_layers and j > 0:
                        break
                    if i == self.hidden_layers:
                        self.sigma[i, j] = (self.o[i, j]-y)*self.o[i, j]*(1-self.o[i, j])
                    else:
                        self.sigma[i, j] = self.sigma[i+1].dot(self.w[i+1][:,j]) * self.o[i, j] * (1 - self.o[i, j])
                    for k in range(self.neurons_per_layer):
                        if i == 0:
                            self.last_delta_w[i, j, k] = inertia*self.last_delta_w[i, j, k] + (1-inertia)*learning_rate * x[k] * self.sigma[i, j]
                            self.w[i, j, k] -= self.last_delta_w[i, j, k]
                        else:
                            self.last_delta_w[i, j, k] = inertia*self.last_delta_w[i, j, k] + (1-inertia)*learning_rate * self.o[i-1, k] * self.sigma[i, j]
                            self.w[i, j, k] -= self.last_delta_w[i, j, k]

def main():
    print('iterations: ', iterations)
    print('step_size:' , step_size)
    print('inertia:', inertia)
    mp = multilayer_perceptron(1, 3)
    dataset = train_ds
    N = len(dataset)
    p = 3
    y = dataset[:,2:3]
    X = np.append(np.ones([N, 1]), dataset[:,0:p-1], axis=1)
    test_X = np.append(np.ones([len(test_ds), 1]), test_ds[:,0:p-1], axis=1)
    test_y = test_ds[:,2:3]
    for iteration in range(iterations):
        mp.backpropagation(X, y, step_size)
    print(mp)
    mp.test_accuracy(test_X, test_y)

    show_samples(train_ds)
    mp.show_decision_boundary(-15, 25, -10, 15)
    plt.show()
    input('Press enter to continue..')

main()

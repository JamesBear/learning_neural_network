
from multilayer_perceptron import *
inertia = 0.0
iterations = 10000
learning_rate = 1

def visualize_xor(mp):

    x1 = np.array([0, 1])
    x2 = np.array([0, 1])
    for i in x1:
        for j in x2:
            if i != j:
                cross = plt.scatter(i, j, marker='+', color='r')
            else:
                dot = plt.scatter(i, j, marker='o', color='b')

    plt.title('Defining XOR using backpropagation')
    plt.legend([cross, dot], ['True', 'False'])

    mp.show_decision_boundary(-1, 2, -1, 2)

    plt.show()


print('iterations: ', iterations)
print('learning_rate:', learning_rate)
print('inertia:', inertia)
initial_weights = np.array([0])
mp = multilayer_perceptron(4, 3, random_scale = -3)
print('layers*neurons: {}*{}'.format(mp.hidden_layers, mp.neurons_per_layer))
X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
Y = np.array([0, 1, 0, 1])
for iteration in range(iterations):
    mp.backpropagation(X, Y, learning_rate)

mp.test_accuracy(X, Y, debug=True)
print(mp)
visualize_xor(mp)
#visualize_xor()

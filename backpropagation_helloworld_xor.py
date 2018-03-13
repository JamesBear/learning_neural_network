
from multilayer_perceptron import *

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
mp = multilayer_perceptron(1, 3)
for iteration in range(iterations):
    mp.backpropagation(np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]]), np.array([0, 1, 0, 1]), learning_rate)

mp.test_accuracy(np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]]), np.array([0, 1, 0, 1]), debug=True)
print(mp)
visualize_xor(mp)
#visualize_xor()

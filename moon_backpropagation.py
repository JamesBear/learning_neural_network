

from multilayer_perceptron import *

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



def main():
    print('iterations: ', iterations)
    print('step_size:' , step_size)
    print('inertia:', inertia)
    mp = multilayer_perceptron(1, 3)
    dataset = train_ds
    N = len(dataset)
    p = 3
    y = dataset[:,2:3]
    X = dataset[:,0:p-1]
    test_X = test_ds[:,0:p-1]
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

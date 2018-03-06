
import matplotlib.pyplot as plt
import numpy as np

_lambda = 10
TRAIN_RATIO_INVERSE = 3 # inverse of 1/3 is 3
file_name = 'moon_dataset_pairs3000_r10_w6_d0.npy'
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


def run_least_squares(dataset):
    m = len(dataset)
    y = dataset[:,2:3]
    X = np.append(np.ones([m, 1]), dataset[:,0:2], axis=1)

    try:
        inv = np.linalg.inv(X.T.dot(X) + _lambda*np.identity(3))
    except np.linalg.LinAlgError:
        print('Ah-oh, cannot invert this one!')
    else:
        pass

    beta = inv.dot(X.T).dot(y)

    draw_decision_boundary(beta)

    return beta

def main():
    show_samples(train_ds)
    beta = run_least_squares(train_ds)
    plt.show()
    score(beta, test_ds)
    input('Press enter to continue..')

main()

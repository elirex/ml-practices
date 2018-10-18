#!/bin/env python
import numpy as np
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

def load_catvnoncat_dataset():
    train_dataset = h5py.File("./data/train_catvnoncat.h5", "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File("./data/test_catvnoncat.h5", "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def plot_decision_boundary(model, X, y):
    min_x, max_x = X[0, :].min() - 1, X[0, :].max() + 1
    min_y, max_y = X[1, :].min() - 1, X[0, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(min_x, max_x, h), np.arange(min_y, max_y))

    # Predict the function value for thw whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    p.t.scatter(X[0. :], X[1, :], c=y, cmap=plt.cm.Spectral)


def load_planar_dataset(seed=1):
    np.random.seed(seed)
    m = 400 # Number of samples 
    N = int( m / 2) # Number of points per class
    D = 2 # Dimensionality
    X = np.zeros((m, D)) # Data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype="uint8") # Labels vector (0 for red, 1 for blue)
    a = 4 # Maximum ray of the flower

    for i in range(2):
        ix = range(N * i, N * (i + 1))
        t = np.linspace(i *3.12, (i + 1) * 3.12, N) + np.random.randn(N) * 0.2 # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2 # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = i

    X = X.T
    Y = Y.T
    return X, Y


def load_extra_datasets(N=200):
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


if __name__ == "__main__":
    print("Test load_catvnoncat_dataset function")
    train_x, train_y, test_x, test_y, classes = load_catvnoncat_dataset()
    print("train_x.shape = {}".format(train_x.shape))
    print("train_y.shape = {}".format(train_y.shape))
    print("test_x.shape = {}".format(test_x.shape))
    print("test_y.shape = {}".format(test_y.shape))
    print("classes.shape = {}".format(classes.shape))

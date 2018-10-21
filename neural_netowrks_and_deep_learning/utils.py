#!/bin/env python
import numpy as np
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Arguments:
    z - A numpy array of any size.
    Returns:
    s - sigmoid(z)
    """
    s = 1.0 / (1 + np.exp(-z))
    return s

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


if __name__ == "__main__":
    print("Test load_catvnoncat_dataset function")
    train_x, train_y, test_x, test_y, classes = load_catvnoncat_dataset()
    print("train_x.shape = {}".format(train_x.shape))
    print("train_y.shape = {}".format(train_y.shape))
    print("test_x.shape = {}".format(test_x.shape))
    print("test_y.shape = {}".format(test_y.shape))
    print("classes.shape = {}".format(classes.shape))

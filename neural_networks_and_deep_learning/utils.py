import os
import numpy as np
import h5py

def print_accuracy(set_type, Y, predicted_Y):
    print("{} accuracy: {}".format(set_type, 100 - np.mean(np.abs(predicted_Y -Y)) * 100))


def sigmoid(z):
    """
    Arguments:
    z - A numpy array of any size.
    Returns:
    s - sigmoid(z)
    """
    s = 1.0 / (1 + np.exp(-z))
    return s


def load_catvnoncat_dataset(path):
    train_path = os.path.join(path, "train_catvnoncat.h5")
    test_path = os.path.join(path, "test_catvnoncat.h5")

    train_dataset = h5py.File(train_path, "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File(test_path, "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes

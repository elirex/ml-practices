import numpy as np

def sigmoid(Z):
    """
    Arguments: 
    Z - numpy array of any shape

    Returns:
    A - acatived by sigmoid, same shape as Z 
    cache - stored 'Z' for computing backpropagation
    """
    A  = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_derivative(dA, cache):
    """
    Arguments:
    dA - activated gradient of any shape
    cache - 'Z' where computing backward propagation efficiently

    Returns:
    dZ - gradient of the cost
    """
    Z = cache
    S = 1 / (1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ


def relu(Z):
    """
    Arguments:
    Z - numpy array of any shape

    Returns:
    A - acatived by relu, same shape as Z
    cache - stored 'Z' for computing backpropagation
    """

    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_derivative(dA, cache):
    """
    Arguments:
    dA - activated gradient of any shape
    cache - 'Z' where computing backward propagation efficiently

    Returns:
    dZ - gradient of the cost
    """

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

import numpy as np
import math
import tensorflow as tf


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """ Arguments:
    X - Input data, shape is (m, Hi, Wi, Ci)
    Y - True "label" vector, shape (1, number of samples)
    mini_batch_size - size of the mini-batches
    seed - numpy random seed

    Returns:
    mini_batches - list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size) # Number of mini batches

    for k in range(0, num_complete_minibatches):
        batch_start = k * mini_batch_size
        batch_end = k * mini_batch_size + mini_batch_size
        mini_batch_X = shuffled_X[batch_start : batch_end, :, :, :]
        mini_batch_Y = shuffled_Y[batch_start : batch_end, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini_batch < mini_batch_size)
    if m % mini_batch_size != 0:
        batch_start = num_complete_minibatches * mini_batch_size
        batch_end = m
        mini_batch_X = shuffled_X[batch_start : batch_end, :, :, :]
        mini_batch_Y = shuffled_Y[batch_start : batch_end, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batchs


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, params):
    """
    Arguments:
    X - data type is tensorflow placeholder, input dataset,shape is (the dimination of input data, number of samples)
    params - data type is python dicitionary, containing forward parameters
        "W1", "b1", "W2", "b2", "W3", "b3".
    
    Returns:
    Z3 - the output of the last LINEAR unit
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


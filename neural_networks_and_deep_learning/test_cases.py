import numpy as np

def test_case_layer_sizes():
    np.random.seed(1)
    mock_X = np.random.randn(5, 3)
    mock_Y = np.random.randn(2, 3)
    return mock_X, mock_Y


def test_case_initialize_parameters():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y


def test_case_forward_progagation():
    np.random.seed(1)
    mock_x = np.random.randn(2, 3)

    mock_params = {"W1": np.random.randn(4, 2),
            "b1": np.random.randn(4, 1),
            "W2": np.random.randn(1, 4),
            "b2": np.random.randn(1, 1)}
    return mock_x, mock_params


def test_case_compute_cost():
    np.random.seed(1)
    mock_params = {"W1": np.random.randn(4, 2),
            "b1": np.random.randn(4, 1),
            "W2": np.random.randn(1, 4),
            "b2": np.random.randn(1, 1)}

    mock_Y = (np.random.randn(1, 3) > 0)
    mock_A2 = np.random.uniform(low=0., high=1., size=(1, 3))
    return mock_A2, mock_Y, mock_params


def test_case_backward_progagation():
    np.random.seed(1)
    mock_X = np.random.randn(2, 3)
    mock_Y = (np.random.rand(1, 3) > 0)

    mock_params = {"W1": np.random.randn(4, 2),
            "b1": np.random.randn(4, 1),
            "W2": np.random.randn(1, 4),
            "b2": np.random.randn(1, 1)}

    mock_cache = {"A1": np.random.randn(4, 3),
            "Z1": np.random.randn(4, 3),
            "A2": np.random.uniform(0., 1., (1, 3)),
            "Z2": np.random.randn(1, 3)}

    return mock_params, mock_cache, mock_X, mock_Y


def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A, W, b


def linear_activation_forward_test_case():
    np.random.seed(1)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def deep_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters


def compute_cost_test_case():
    np.random.seed(1)
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])
    return Y, aL


def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)
    return dZ, linear_cache


def linear_activation_backward_test_case():
    np.random.seed(1)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    return dA, linear_activation_cache


def deep_model_backward_test_case():
    np.random.seed(1)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches


def update_parameters_test_case():
    np.random.seed(1)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads


def deep_model_forward_test_case_2hidden():
    np.random.seed(1)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return X, parameters


def deep_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters


def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])
    return Y, aL



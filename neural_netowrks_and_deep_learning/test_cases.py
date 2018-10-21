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



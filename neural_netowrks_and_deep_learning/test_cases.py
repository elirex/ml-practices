import numpy as np

def test_case_layer_sizes():
    np.random.seed(1)
    mock_X = np.random.randn(5, 3)
    mock_Y = np.random.randn(2, 3)
    return mock_X, mock_Y


def test_case_initialize_parameters():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y

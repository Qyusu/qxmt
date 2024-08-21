import numpy as np

from qxmt.datasets.dummy import generate_linear_separable_data


def test_generate_linear_separable_data() -> None:
    X, y = generate_linear_separable_data(n_samples=100, n_features=2, noise=0.1, scale=1.0)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert np.unique(y).tolist() == [-1, 1]

    # high dimension
    X, y = generate_linear_separable_data(n_samples=100, n_features=10, noise=0.1, scale=1.0)
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert np.unique(y).tolist() == [-1, 1]

    # up scale
    X, y = generate_linear_separable_data(n_samples=1000, n_features=2, noise=0.1, scale=10.0)
    assert X.shape == (1000, 2)
    assert X.max() >= 10.0
    assert y.shape == (1000,)
    assert np.unique(y).tolist() == [-1, 1]

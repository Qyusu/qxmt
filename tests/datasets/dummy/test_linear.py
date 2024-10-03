import numpy as np
import pytest

from qxmt.datasets.dummy.linear import generate_linear_separable_data


@pytest.mark.parametrize(
    "n_samples, n_features, noise, scale, use_positive_labels",
    [
        (100, 2, 0.1, 1.0, True),  # default case
        (100, 10, 0.1, 1.0, True),  # high dimension
        (1000, 2, 0.1, 10.0, True),  # up scale
        (100, 2, 0.1, 1.0, False),  # allow negative labels
    ],
)
def test_generate_linear_separable_data(
    n_samples: int, n_features: int, noise: float, scale: float, use_positive_labels: bool
) -> None:
    X, y = generate_linear_separable_data(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        scale=scale,
        use_positive_labels=use_positive_labels,
    )
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert np.unique(y).tolist() == [0, 1] if use_positive_labels else [-1, 1]

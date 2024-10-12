import numpy as np
import pytest

from qxmt.datasets.dummy.linear import generate_linear_separable_data


@pytest.mark.parametrize(
    "n_samples, n_features, n_classes, noise, scale",
    [
        (100, 2, 2, 0.1, 1.0),  # binary class
        (100, 2, 5, 0.1, 1.0),  # multi class
        (100, 10, 2, 0.1, 1.0),  # high dimension
        (1000, 2, 2, 0.1, 1.0),  # up sample
    ],
)
def test_generate_linear_separable_data(
    n_samples: int, n_features: int, n_classes: int, noise: float, scale: float
) -> None:
    X, y = generate_linear_separable_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        noise=noise,
        scale=scale,
    )
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)

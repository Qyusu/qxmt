import numpy as np
import pytest

from qxmt.datasets.dummy.linear import (
    generate_linear_regression_data,
    generate_linear_separable_data,
)


@pytest.mark.parametrize(
    "n_samples, n_features, n_classes, noise, scale",
    [
        pytest.param(100, 2, 2, 0.1, 1.0, id="binary class"),
        pytest.param(100, 2, 5, 0.1, 1.0, id="multi class"),
        pytest.param(100, 10, 2, 0.1, 1.0, id="high dimension"),
        pytest.param(1000, 2, 2, 0.1, 1.0, id="up sample"),
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


@pytest.mark.parametrize(
    "n_samples, n_features, noise, scale",
    [
        pytest.param(100, 2, 0.1, 1.0, id="default"),
        pytest.param(100, 5, 0.1, 1.0, id="high dimension"),
        pytest.param(1000, 2, 0.1, 1.0, id="up sample"),
    ],
)
def test_generate_linear_regression_data(n_samples: int, n_features: int, noise: float, scale: float) -> None:
    X, y = generate_linear_regression_data(n_samples=n_samples, n_features=n_features, noise=noise, scale=scale)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)

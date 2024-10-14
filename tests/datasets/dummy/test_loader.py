import numpy as np
from pytest_mock import MockerFixture

from qxmt.datasets.dummy import load_dummy_dataset


def test_load_dummy_dataset(mocker: MockerFixture) -> None:

    X, y = load_dummy_dataset(
        task_type="classification",
        generate_method="linear",
        random_seed=0,
        params={"n_samples": 100, "n_features": 2, "n_classes": 2},
    )
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert len(np.unique(y)) <= 2

    X, y = load_dummy_dataset(
        task_type="regression",
        generate_method="linear",
        random_seed=0,
        params={"n_samples": 5, "n_features": 1},
    )
    assert X.shape == (5, 1)
    assert y.shape == (5,)

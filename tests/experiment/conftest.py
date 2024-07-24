from pathlib import Path

import numpy as np
import pytest

from quri import Experiment
from quri.datasets.schema import Dataset
from quri.models.base_model import BaseModel
from quri.models.qsvm import QSVM


@pytest.fixture(scope="function")
def base_experiment(tmp_path: Path) -> Experiment:
    return Experiment(
        name="test_exp",
        desc="test experiment",
        root_experiment_dirc=tmp_path,
    )


@pytest.fixture(scope="function")
def base_model() -> BaseModel:
    return QSVM()


@pytest.fixture(scope="function")
def dataset() -> Dataset:
    return Dataset(
        x_train=np.random.rand(10, 10),
        y_train=np.random.randint(2, size=10),
        x_test=np.random.rand(10, 10),
        y_test=np.random.randint(2, size=10),
        features=["feature_1", "feature_2"],
    )

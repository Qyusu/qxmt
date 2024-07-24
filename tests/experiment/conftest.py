from pathlib import Path

import pytest

from qk_manager import Experiment
from qk_manager.models.base_kernel_model import BaseKernelModel
from qk_manager.models.qsvm import QSVM


@pytest.fixture(scope="function")
def base_experiment(tmp_path: Path) -> Experiment:
    return Experiment(
        name="test_exp",
        desc="test experiment",
        root_experiment_dirc=tmp_path,
    )


@pytest.fixture(scope="function")
def base_model() -> BaseKernelModel:
    return QSVM()

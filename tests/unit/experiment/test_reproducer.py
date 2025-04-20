from typing import Callable

import pytest
from pytest_mock import MockFixture

from qxmt import Experiment
from qxmt.configs import ExperimentConfig
from qxmt.datasets import Dataset
from qxmt.exceptions import ExperimentNotInitializedError, ReproductionError
from qxmt.models.qkernels import BaseMLModel


class TestReproducer:
    def test_reproduce_qkernel(
        self,
        mocker: MockFixture,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_model: BaseMLModel,
        qkernel_experiment_config: ExperimentConfig,
    ) -> None:
        # experiment not initialized.
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.reproduce(run_id=1)

        # experiment initialization and execution.
        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            n_jobs=1,
        )

        # Instance mode not supported for reproduction.
        with pytest.raises(ReproductionError):
            base_experiment.reproduce(run_id=1)

        # run_id=2 not exist.
        with pytest.raises(ValueError):
            base_experiment.reproduce(run_id=2)

        # run_id=3 executed from config file.
        mocker.patch("qxmt.datasets.DatasetBuilder.build", return_value=dataset)
        mocker.patch("qxmt.models.qkernels.KernelModelBuilder.build", return_value=state_vec_model)
        base_experiment.run(config_source=qkernel_experiment_config, n_jobs=1)

        # reproduce execution.
        artifact, record = base_experiment.reproduce(run_id=2)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)
        assert record.run_id == 2

    def test_reproduce_vqe(self, mocker: MockFixture, base_experiment: Experiment) -> None:
        pass

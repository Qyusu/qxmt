from typing import Callable

import pytest
from pytest_mock import MockFixture

from qxmt import Experiment
from qxmt.configs import ExperimentConfig
from qxmt.datasets import Dataset
from qxmt.exceptions import ExperimentNotInitializedError, ReproductionError
from qxmt.models.qkernels import BaseMLModel
from qxmt.models.vqe import BaseVQE


class TestReproducer:
    def test_reproduce_setting(self, base_experiment: Experiment) -> None:
        # experiment not initialized.
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.reproduce(run_id=1)

    def test_reproduce_instance_mode(
        self,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_qkernel_model: BaseMLModel,
    ) -> None:
        # experiment initialization and execution.
        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            n_jobs=1,
        )

        # Instance mode not supported for reproduction.
        with pytest.raises(ReproductionError):
            base_experiment.reproduce(run_id=1)

    def test_reproduce_not_exist_run_id(self, base_experiment: Experiment) -> None:
        # experiment initialization and execution.
        base_experiment.init()

        # run_id=2 not exist.
        with pytest.raises(ValueError):
            base_experiment.reproduce(run_id=1)

    def test_reproduce_qkernel(
        self,
        mocker: MockFixture,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_qkernel_model: BaseMLModel,
        qkernel_experiment_config: ExperimentConfig,
    ) -> None:
        # experiment initialization and execution.
        base_experiment.init()

        # execute from config file.
        mocker.patch(
            "qxmt.datasets.DatasetBuilder.build",
            return_value=create_random_dataset(data_num=100, feature_num=5, class_num=2),
        )
        mocker.patch("qxmt.models.qkernels.KernelModelBuilder.build", return_value=state_vec_qkernel_model)
        base_experiment.run(config_source=qkernel_experiment_config, n_jobs=1)

        # reproduce execution.
        artifact, record = base_experiment.reproduce(run_id=1)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)
        assert record.run_id == 1

    def test_reproduce_vqe(
        self,
        mocker: MockFixture,
        base_experiment: Experiment,
        state_vec_vqe_model: BaseVQE,
        vqe_experiment_config: ExperimentConfig,
    ) -> None:
        # experiment initialization and execution.
        base_experiment.init()

        # execute from config file.
        mocker.patch("qxmt.models.vqe.VQEModelBuilder.build", return_value=state_vec_vqe_model)
        base_experiment.run(config_source=vqe_experiment_config, n_jobs=1)

        # reproduce execution.
        artifact, record = base_experiment.reproduce(run_id=1)
        assert isinstance(artifact.model, BaseVQE)
        assert record.run_id == 1

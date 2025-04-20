from pathlib import Path
from typing import Callable

import pytest
from pytest_mock import MockFixture

from qxmt import Experiment
from qxmt.configs import ExperimentConfig
from qxmt.constants import (
    DEFAULT_EXP_CONFIG_FILE,
    DEFAULT_MODEL_NAME,
    DEFAULT_SHOT_RESULTS_NAME,
)
from qxmt.datasets import Dataset
from qxmt.exceptions import ExperimentNotInitializedError
from qxmt.experiment.executor import QKernelExecutor, VQEExecutor
from qxmt.experiment.schema import RunArtifact, RunRecord, RunTime, VQERunTime
from qxmt.models.qkernels import BaseMLModel
from qxmt.models.vqe import BaseVQE
from qxmt.utils import save_experiment_config_to_yaml


class TestQKernelExecutor:
    def test_run_from_config(
        self,
        mocker: MockFixture,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_model: BaseMLModel,
        qkernel_experiment_config: ExperimentConfig,
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        mocker.patch("qxmt.datasets.DatasetBuilder.build", return_value=dataset)
        mocker.patch("qxmt.models.qkernels.KernelModelBuilder.build", return_value=state_vec_model)

        executor = QKernelExecutor(experiment=base_experiment)
        artifact, record = executor.run_from_config(
            config=qkernel_experiment_config,
            commit_id="test_commit",
            run_dirc=Path("/tmp"),
            n_jobs=1,
            repo_path=None,
            show_progress=False,
            add_results=True,
        )

        assert isinstance(artifact, RunArtifact)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)
        assert isinstance(record, RunRecord)
        assert record.commit_id == "test_commit"
        assert record.config_file_name == Path(DEFAULT_EXP_CONFIG_FILE)
        assert isinstance(record.runtime, RunTime)
        assert record.runtime.train_seconds is not None
        assert record.runtime.validation_seconds is not None
        assert record.runtime.test_seconds is not None

    def test_run_from_instance(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_model: BaseMLModel
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        executor = QKernelExecutor(experiment=base_experiment)
        artifact, record = executor.run_from_instance(
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            save_shots_path=None,
            save_model_path=Path("/tmp"),
            default_metrics_name=["accuracy"],
            custom_metrics=[],
            desc="Test QKernel run",
            commit_id="test_commit",
            config_file_name=Path(DEFAULT_EXP_CONFIG_FILE),
            repo_path=None,
            add_results=False,
        )

        assert isinstance(artifact, RunArtifact)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)
        assert isinstance(record, RunRecord)
        assert record.commit_id == "test_commit"
        assert record.config_file_name == Path(DEFAULT_EXP_CONFIG_FILE)
        assert isinstance(record.runtime, RunTime)
        assert record.runtime.train_seconds is not None
        assert record.runtime.validation_seconds is not None
        assert record.runtime.test_seconds is not None


class TestVQEExecutor:
    def test_run_from_config(self, mock_experiment, mock_config, mock_model, mocker):
        pass

    def test_run_from_instance(self, mock_experiment, mock_model):
        pass

    def test_run_from_instance_remote_error(self, mock_experiment, mock_model):
        pass

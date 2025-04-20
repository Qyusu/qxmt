from pathlib import Path
from typing import Callable

from pytest_mock import MockFixture

from qxmt import Experiment
from qxmt.ansatze.pennylane.uccsd import UCCSDAnsatz
from qxmt.configs import ExperimentConfig
from qxmt.constants import DEFAULT_EXP_CONFIG_FILE
from qxmt.datasets import Dataset
from qxmt.experiment.executor import QKernelExecutor, VQEExecutor
from qxmt.experiment.schema import RunArtifact, RunRecord, RunTime, VQERunTime
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
from qxmt.models.qkernels import BaseMLModel
from qxmt.models.vqe import BaseVQE


class TestQKernelExecutor:
    # [TODO]: add test for shots_qkernel_model by parameterizing
    def test_run_from_config(
        self,
        mocker: MockFixture,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_qkernel_model: BaseMLModel,
        qkernel_experiment_config: ExperimentConfig,
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        mocker.patch("qxmt.datasets.DatasetBuilder.build", return_value=dataset)
        mocker.patch("qxmt.models.qkernels.KernelModelBuilder.build", return_value=state_vec_qkernel_model)

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
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        executor = QKernelExecutor(experiment=base_experiment)
        artifact, record = executor.run_from_instance(
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
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
    # [TODO]: add test for shots_vqe_model by parameterizing
    def test_run_from_config(
        self,
        mocker: MockFixture,
        base_experiment: Experiment,
        hamiltonian: MolecularHamiltonian,
        ansatz: UCCSDAnsatz,
        state_vec_vqe_model: BaseVQE,
        vqe_experiment_config: ExperimentConfig,
    ) -> None:
        mocker.patch("qxmt.hamiltonians.builder.HamiltonianBuilder.build", return_value=hamiltonian)
        mocker.patch("qxmt.ansatze.builder.AnsatzBuilder.build", return_value=ansatz)
        mocker.patch("qxmt.models.vqe.VQEModelBuilder.build", return_value=state_vec_vqe_model)
        executor = VQEExecutor(experiment=base_experiment)
        artifact, record = executor.run_from_config(
            config=vqe_experiment_config,
            commit_id="test_commit",
            run_dirc=Path("/tmp"),
            n_jobs=1,
            repo_path=None,
            show_progress=False,
            add_results=True,
        )

        assert isinstance(artifact, RunArtifact)
        assert isinstance(artifact.model, BaseVQE)
        assert artifact.dataset is None
        assert isinstance(record, RunRecord)
        assert record.commit_id == "test_commit"
        assert record.config_file_name == Path(DEFAULT_EXP_CONFIG_FILE)
        assert isinstance(record.runtime, VQERunTime)
        assert record.runtime.optimize_seconds is not None

    def test_run_from_instance(self, base_experiment: Experiment, state_vec_vqe_model: BaseVQE):
        executor = VQEExecutor(experiment=base_experiment)
        artifact, record = executor.run_from_instance(
            model=state_vec_vqe_model,
            save_shots_path=None,
            default_metrics_name=["final_cost", "hf_energy"],
            custom_metrics=[],
            desc="Test VQE run",
            commit_id="test_commit",
            config_file_name=Path(DEFAULT_EXP_CONFIG_FILE),
            repo_path=None,
            add_results=False,
        )

        assert isinstance(artifact, RunArtifact)
        assert isinstance(artifact.model, BaseVQE)
        assert artifact.dataset is None
        assert isinstance(record, RunRecord)
        assert record.commit_id == "test_commit"
        assert record.config_file_name == Path(DEFAULT_EXP_CONFIG_FILE)
        assert isinstance(record.runtime, VQERunTime)
        assert record.runtime.optimize_seconds is not None

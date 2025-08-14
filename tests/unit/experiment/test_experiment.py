import json
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
from pytest_mock import MockFixture

from qxmt import Experiment
from qxmt.configs import ExperimentConfig
from qxmt.constants import DEFAULT_EXP_DB_FILE
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    ExperimentSettingError,
)
from qxmt.experiment.schema import RunArtifact, RunRecord
from qxmt.models.qkernels import BaseMLModel
from qxmt.models.vqe import BaseVQE
from qxmt.utils import save_experiment_config_to_yaml


class TestExperimentSettings:
    def test_default(self, tmp_path: Path) -> None:
        default_exp = Experiment(root_experiment_dirc=tmp_path).init()
        assert isinstance(default_exp.name, str)
        assert default_exp.desc == ""
        assert default_exp.current_run_id == 0
        assert default_exp.experiment_dirc.exists()

    def test__is_initialized(self, base_experiment: Experiment) -> None:
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment._is_initialized()

        base_experiment.init()
        assert base_experiment._is_initialized

    def test_initialization(self, base_experiment: Experiment) -> None:
        base_experiment.init()
        assert base_experiment.name == "test_exp"
        assert base_experiment.desc == "test experiment"
        assert base_experiment.current_run_id == 0
        assert base_experiment.experiment_dirc.exists()

    def test_generate_default_name(self) -> None:
        assert len(Experiment._generate_default_name()) == 20


class TestExperimentInit:
    def test_init_with_default_settings(self, tmp_path: Path) -> None:
        exp = Experiment(root_experiment_dirc=tmp_path).init()
        assert exp.exp_db is not None
        assert isinstance(exp.exp_db.name, str)
        assert exp.exp_db.desc == ""
        assert exp.exp_db.working_dirc == Path.cwd()
        assert exp.exp_db.experiment_dirc == tmp_path / exp.name  # type: ignore
        assert exp.exp_db.runs == []
        assert exp.experiment_dirc.exists()

    def test_init_with_custom_settings(self, base_experiment: Experiment, tmp_path: Path) -> None:
        base_experiment.init()
        assert base_experiment.exp_db is not None
        assert base_experiment.exp_db.name == "test_exp"
        assert base_experiment.exp_db.desc == "test experiment"
        assert base_experiment.exp_db.working_dirc == Path.cwd()
        assert base_experiment.exp_db.experiment_dirc == tmp_path / "test_exp"
        assert base_experiment.exp_db.runs == []
        assert base_experiment.experiment_dirc.exists()

    def test_init_with_existing_directory(self, tmp_path: Path) -> None:
        exp = Experiment(name="existing_exp", root_experiment_dirc=tmp_path).init()
        assert exp.experiment_dirc.exists()

        (exp.experiment_dirc / "dummy.txt").touch()
        with pytest.raises(ExperimentSettingError) as exc_info:
            _ = Experiment(name="existing_exp", root_experiment_dirc=tmp_path).init()

        assert str(exc_info.value) == f"Experiment directory '{exp.experiment_dirc}' already exists."


class TestLoadExperiment:
    # [TODO]: mock repository.load method
    def set_dummy_experiment_data(self, experiment_dirc: Path) -> None:
        dummy_experiment_dict = {
            "name": "load_exp",
            "desc": "load experiment",
            "working_dirc": str(experiment_dirc),
            "experiment_dirc": str(experiment_dirc / "load_exp"),
            "runs": [
                {
                    "run_id": 1,
                    "desc": "",
                    "execution_time": "2024-07-24 17:33:55.305025 JST+0900",
                    "runtime": {"train_seconds": 7.42, "validation_seconds": 1.23, "test_seconds": 1.36},
                    "commit_id": "commit_1",
                    "config_file_name": "config.yaml",
                    "evaluations": {
                        "validation": None,
                        "test": {
                            "accuracy": 0.65,
                            "precision": 0.69,
                            "recall": 0.65,
                            "f1_score": 0.67,
                        },
                    },
                },
                {
                    "run_id": 2,
                    "desc": "",
                    "execution_time": "2024-07-24 17:34:01.932990 JST+0900",
                    "runtime": {"train_seconds": 120.5, "validation_seconds": None, "test_seconds": 33.3},
                    "commit_id": "commit_2",
                    "config_file_name": "",
                    "evaluations": {
                        "validation": {
                            "accuracy": 0.6,
                            "precision": 0.68,
                            "recall": 0.62,
                            "f1_score": 0.65,
                        },
                        "test": {
                            "accuracy": 0.5,
                            "precision": 0.47,
                            "recall": 0.55,
                            "f1_score": 0.50,
                        },
                    },
                },
            ],
        }

        exp_json_path = Path(f"{dummy_experiment_dict['experiment_dirc']}/{DEFAULT_EXP_DB_FILE}")
        exp_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exp_json_path, "w") as json_file:
            json.dump(dummy_experiment_dict, json_file, indent=4)

    def test_load(self, tmp_path: Path) -> None:
        self.set_dummy_experiment_data(tmp_path)

        # error pattern 1: experiment file does not exist
        with pytest.raises(FileNotFoundError) as exc_info:
            Experiment(root_experiment_dirc=tmp_path).load(exp_dirc=tmp_path / "not_exist_exp")
        assert str(exc_info.value) == f"{tmp_path}/not_exist_exp/{DEFAULT_EXP_DB_FILE} does not exist."

        # error pattern 2: experiment directory does not exist
        with pytest.raises(ExperimentSettingError) as exc_info:
            Experiment(name="not_exist", root_experiment_dirc=tmp_path).load(exp_dirc=tmp_path / "load_exp")
        assert str(exc_info.value) == f"Experiment directory '{tmp_path}/not_exist' does not exist."

        # pattern 1: load default settings
        loaded_exp = Experiment(root_experiment_dirc=tmp_path).load(exp_dirc=tmp_path / "load_exp")
        assert loaded_exp.name == "load_exp"
        assert loaded_exp.desc == "load experiment"
        assert loaded_exp.root_experiment_dirc == tmp_path
        assert loaded_exp.experiment_dirc == tmp_path / "load_exp"
        assert loaded_exp.current_run_id == 2
        assert len(loaded_exp.exp_db.runs) == 2  # type: ignore

        # pattern 2: confirm updated settings
        (tmp_path / "update_exp").mkdir(parents=True, exist_ok=True)
        updated_exp = Experiment(name="update_exp", desc="update experiment", root_experiment_dirc=tmp_path).load(
            exp_dirc=tmp_path / "load_exp"
        )

        # confirm updated settings
        assert updated_exp.name == "update_exp"
        assert updated_exp.desc == "update experiment"
        assert updated_exp.root_experiment_dirc == tmp_path
        assert updated_exp.experiment_dirc == tmp_path / "update_exp"
        assert updated_exp.current_run_id == 2
        assert updated_exp.exp_db.working_dirc == Path.cwd()  # type: ignore
        assert len(updated_exp.exp_db.runs) == 2  # type: ignore

        # confirm experiment data is saved
        assert (updated_exp.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()


class TestExperimentRun:
    def test__run_setup(self, base_experiment: Experiment) -> None:
        base_experiment.init()
        assert base_experiment.current_run_id == 0
        assert not base_experiment.experiment_dirc.joinpath("run_1").exists()

        current_run_dirc = base_experiment._run_setup()
        assert base_experiment.current_run_id == 1
        assert current_run_dirc == base_experiment.experiment_dirc / "run_1"

    def test__run_backfill(self, base_experiment: Experiment) -> None:
        exp = base_experiment.init()
        exp.current_run_id = 1

        run_dirc = exp.experiment_dirc / "run_1"
        run_dirc.mkdir()

        exp._run_backfill()
        assert not run_dirc.exists()
        assert exp.current_run_id == 0

    def test_run_with_config(
        self, mocker: MockFixture, tmp_path: Path, qkernel_experiment_config: ExperimentConfig
    ) -> None:
        mock_qkernel_executor = mocker.patch("qxmt.experiment.experiment.QKernelExecutor")
        mock_executor_instance = mock_qkernel_executor.return_value
        mock_artifact = mocker.Mock(spec=RunArtifact)
        mock_record = mocker.Mock(spec=RunRecord)
        mock_executor_instance.run_from_config.return_value = (mock_artifact, mock_record)

        exp = Experiment(name="test_exp", root_experiment_dirc=tmp_path).init()
        config_path = tmp_path / "config.yaml"
        save_experiment_config_to_yaml(qkernel_experiment_config, config_path)
        artifact, record = exp.run(config_source=config_path)

        assert mock_qkernel_executor.called
        mock_executor_instance.run_from_config.assert_called_once()
        assert artifact == mock_artifact
        assert record == mock_record
        assert exp.current_run_id == 1
        assert len(exp.exp_db.runs) == 1  # type: ignore

    def test_run_with_instance(self, mocker: MockFixture, tmp_path: Path, create_random_dataset: Callable) -> None:
        mock_qkernel_executor = mocker.patch("qxmt.experiment.experiment.QKernelExecutor")
        mock_executor_instance = mock_qkernel_executor.return_value
        mock_artifact = mocker.Mock(spec=RunArtifact)
        mock_record = mocker.Mock(spec=RunRecord)
        mock_executor_instance.run_from_instance.return_value = (mock_artifact, mock_record)

        exp = Experiment(name="test_exp", root_experiment_dirc=tmp_path).init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        model = mocker.Mock(spec=BaseMLModel)
        artifact, record = exp.run(model_type="qkernel", task_type="classification", dataset=dataset, model=model)

        assert mock_qkernel_executor.called
        mock_executor_instance.run_from_instance.assert_called_once()
        assert artifact == mock_artifact
        assert record == mock_record
        assert exp.current_run_id == 1
        assert len(exp.exp_db.runs) == 1  # type: ignore

    def test_run_missing_parameters(self, mocker: MockFixture, tmp_path: Path) -> None:
        exp = Experiment(name="test_exp", root_experiment_dirc=tmp_path).init()

        # not set model_type
        with pytest.raises(ExperimentRunSettingError):
            exp.run(dataset=mocker.Mock(), model=mocker.Mock())

        # not set dataset
        with pytest.raises(ExperimentRunSettingError):
            exp.run(model_type="qkernel", model=mocker.Mock())

        # not set model
        with pytest.raises(ExperimentRunSettingError):
            exp.run(model_type="qkernel", dataset=mocker.Mock())


class TestExperimentResults:
    def test_runs_to_dataframe_empty(self, base_experiment: Experiment) -> None:
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.runs_to_dataframe()

        base_experiment.init()
        df = base_experiment.runs_to_dataframe()
        assert df.empty
        assert isinstance(df, pd.DataFrame)

    def test_runs_to_dataframe_qkernel(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        default_metrics_name = ["accuracy", "precision", "recall", "f1_score"]
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        df = base_experiment.runs_to_dataframe()

        assert Counter(df.columns) == Counter(["run_id", "accuracy", "precision", "recall", "f1_score"])
        assert len(df) == 2
        assert all(df["run_id"] == [1, 2])

    def test_runs_to_dataframe_qkernel_with_validation(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.runs_to_dataframe(include_validation=True)

        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2, include_validation=True)
        default_metrics_name = ["accuracy", "precision", "recall", "f1_score"]
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        df = base_experiment.runs_to_dataframe(include_validation=True)

        assert Counter(df.columns) == Counter(
            [
                "run_id",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "accuracy_validation",
                "precision_validation",
                "recall_validation",
                "f1_score_validation",
            ]
        )
        assert len(df) == 2
        assert all(df["run_id"] == [1, 2])

    def test_runs_to_dataframe_vqe(self, base_experiment: Experiment, state_vec_vqe_model: BaseVQE) -> None:
        base_experiment.init()
        default_metrics_name = ["final_cost", "hf_energy"]
        base_experiment.run(
            model_type="vqe",
            model=state_vec_vqe_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        df = base_experiment.runs_to_dataframe()

        assert Counter(df.columns) == Counter(["run_id", "final_cost", "hf_energy"])
        assert len(df) == 1
        assert df["run_id"].iloc[0] == 1

    def test_runs_to_dataframe_custom_metrics(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        # [TODO]: custom metric receive from BaseMetric class not from config file
        pass

    def test_runs_to_dataframe_invalid_evaluation_type(self, base_experiment: Experiment, mocker: MockFixture) -> None:
        base_experiment.init()
        mock_run = mocker.Mock()
        mock_run.evaluations = "invalid_type"
        base_experiment.exp_db.runs = [mock_run]  # type: ignore

        with pytest.raises(ValueError, match="Invalid run_record.evaluations"):
            base_experiment.runs_to_dataframe()

    def test_get_run_record(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            n_jobs=1,
        )
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_qkernel_model,
            n_jobs=1,
        )

        run_record = base_experiment.get_run_record(base_experiment.exp_db.runs, 1)  # type: ignore
        assert run_record.run_id == 1

        with pytest.raises(ValueError, match="Run record of run_id=3 does not exist."):
            base_experiment.get_run_record(base_experiment.exp_db.runs, 3)  # type: ignore

    def test_save_experiment(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)

        base_experiment.init()
        base_experiment.run(
            model_type="qkernel", task_type="classification", dataset=dataset, model=state_vec_qkernel_model, n_jobs=1
        )
        base_experiment.run(
            model_type="qkernel", task_type="classification", dataset=dataset, model=state_vec_qkernel_model, n_jobs=1
        )
        base_experiment.save_experiment()

        assert (base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()

    def test_save_experiment_custom_filename(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_qkernel_model: BaseMLModel
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        custom_filename = "custom_experiment.json"

        base_experiment.init()
        base_experiment.run(
            model_type="qkernel", task_type="classification", dataset=dataset, model=state_vec_qkernel_model, n_jobs=1
        )
        base_experiment.save_experiment(exp_file=custom_filename)

        assert (base_experiment.experiment_dirc / custom_filename).exists()

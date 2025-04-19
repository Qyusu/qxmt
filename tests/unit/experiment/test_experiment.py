import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from pytest_mock import MockFixture

from qxmt import Experiment
from qxmt.configs import ExperimentConfig
from qxmt.constants import DEFAULT_EXP_DB_FILE
from qxmt.datasets import Dataset
from qxmt.evaluation.metrics import BaseMetric
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    ExperimentSettingError,
    ReproductionError,
)
from qxmt.models.qkernels import BaseMLModel
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


class TestExperimentInit:
    def test_init(self, base_experiment: Experiment, tmp_path: Path) -> None:
        assert base_experiment.exp_db is None

        base_experiment.init()
        exp_db = base_experiment.exp_db
        assert exp_db.name == "test_exp"  # type: ignore
        assert exp_db.desc == "test experiment"  # type: ignore
        assert exp_db.experiment_dirc == tmp_path / "test_exp"  # type: ignore


class TestLoadExperiment:
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

        # Error pattern (not exist experiment file)
        with pytest.raises(FileNotFoundError):
            Experiment(root_experiment_dirc=tmp_path).load(exp_dirc=tmp_path / "not_exist_exp")

        # Error pattern (not exist experiment directory)
        with pytest.raises(ExperimentSettingError):
            Experiment(name="not_exist").load(exp_dirc=tmp_path / "load_exp")

        # Default pattern
        loaded_exp = Experiment(root_experiment_dirc=tmp_path).load(exp_dirc=tmp_path / "load_exp")
        assert loaded_exp.name == "load_exp"
        assert loaded_exp.desc == "load experiment"
        assert loaded_exp.root_experiment_dirc == tmp_path
        assert loaded_exp.experiment_dirc == tmp_path / "load_exp"
        assert loaded_exp.current_run_id == 2
        assert len(loaded_exp.exp_db.runs) == 2  # type: ignore

        # Update Setting Pattern
        (tmp_path / "update_exp").mkdir(parents=True, exist_ok=True)
        updated_exp = Experiment(name="update_exp", desc="update experiment", root_experiment_dirc=tmp_path).load(
            exp_dirc=tmp_path / "load_exp"
        )
        assert updated_exp.name == "update_exp"
        assert updated_exp.desc == "update experiment"
        assert updated_exp.root_experiment_dirc == tmp_path
        assert updated_exp.experiment_dirc == tmp_path / "update_exp"
        assert updated_exp.current_run_id == 2
        assert updated_exp.exp_db.working_dirc == Path.cwd()  # type: ignore
        assert len(updated_exp.exp_db.runs) == 2  # type: ignore


class CustomMetric(BaseMetric):
    def __init__(self, name: str = "custom") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> float:
        score = actual[0] + predicted[0]

        return float(score)


class TestExperimentRun:
    def test__run_setup(self, base_experiment: Experiment) -> None:
        base_experiment.init()
        assert base_experiment.current_run_id == 0
        assert not base_experiment.experiment_dirc.joinpath("run_1").exists()

        base_experiment._run_setup()
        assert base_experiment.current_run_id == 1
        assert base_experiment.experiment_dirc.joinpath("run_1").exists()

    def test__run_backfill(self, base_experiment: Experiment) -> None:
        base_experiment.init()

        with pytest.raises(ExperimentRunSettingError):
            base_experiment.run(dataset=None, model=None, n_jobs=1)

        assert base_experiment.current_run_id == 0
        assert not base_experiment.experiment_dirc.joinpath("run_1").exists()

    def test_run_from_instance(
        self,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_model: BaseMLModel,
        shots_model: BaseMLModel,
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)

        # initialization error check
        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db is None
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.run(dataset=dataset, model=state_vec_model, n_jobs=1)

        # run from dataset and model instance
        base_experiment.init()
        artifact, _ = base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            add_results=True,
            n_jobs=1,
        )
        assert len(base_experiment.exp_db.runs) == 1  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_1/model.pkl").exists()
        assert not base_experiment.experiment_dirc.joinpath("run_1/shots.h5").exists()
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)

        _, _ = base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            add_results=True,
            n_jobs=1,
        )
        _, _ = base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            add_results=True,
            n_jobs=1,
        )
        assert len(base_experiment.exp_db.runs) == 3  # type: ignore

        # run by shots model
        _, _ = base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=shots_model,
            add_results=True,
            n_jobs=1,
        )
        assert base_experiment.experiment_dirc.joinpath("run_4/model.pkl").exists()
        assert base_experiment.experiment_dirc.joinpath("run_4/shots.h5").exists()
        assert len(base_experiment.exp_db.runs) == 4  # type: ignore

        # not add result record (state vector mode)
        _, _ = base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            add_results=False,
            n_jobs=1,
        )
        assert not base_experiment.experiment_dirc.joinpath("run_5/model.pkl").exists()
        assert len(base_experiment.exp_db.runs) == 4  # type: ignore

        # not add result record (shots mode)
        _, _ = base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=shots_model,
            add_results=False,
            n_jobs=1,
        )
        assert not base_experiment.experiment_dirc.joinpath("run_5/model.pkl").exists()
        assert not base_experiment.experiment_dirc.joinpath("run_5/shots.h5").exists()
        assert len(base_experiment.exp_db.runs) == 4  # type: ignore

        # invalid arguments patterm
        with pytest.raises(ExperimentRunSettingError):
            base_experiment.run()

        with pytest.raises(ExperimentRunSettingError):
            base_experiment.run(dataset=dataset)

        with pytest.raises(ExperimentRunSettingError):
            base_experiment.run(model=state_vec_model)

        with pytest.raises(ExperimentRunSettingError):
            base_experiment.run(dataset=dataset, model=state_vec_model)

    def test_run_from_config_state_vec(
        self,
        mocker: MockFixture,
        tmp_path: Path,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        state_vec_model: BaseMLModel,
        qkernel_experiment_config: ExperimentConfig,
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        mocker.patch("qxmt.datasets.DatasetBuilder.build", return_value=dataset)
        mocker.patch("qxmt.models.qkernels.KernelModelBuilder.build", return_value=state_vec_model)

        # initialization error check
        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db is None
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.run(config_source=qkernel_experiment_config, n_jobs=1)

        # run from config instance
        base_experiment.init()
        artifact, _ = base_experiment.run(config_source=qkernel_experiment_config, n_jobs=1)
        assert len(base_experiment.exp_db.runs) == 1  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_1/model.pkl").exists()
        assert base_experiment.experiment_dirc.joinpath("run_1/config.yaml").exists()
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)

        # run from config file
        experiment_config_file = tmp_path / "experiment_config.yaml"
        save_experiment_config_to_yaml(qkernel_experiment_config, experiment_config_file, delete_source_path=True)
        artifact, _ = base_experiment.run(config_source=experiment_config_file, add_results=True, n_jobs=1)
        assert len(base_experiment.exp_db.runs) == 2  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_2/model.pkl").exists()
        assert base_experiment.experiment_dirc.joinpath("run_2/config.yaml").exists()
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)

    def test_run_from_config_shots(
        self,
        mocker: MockFixture,
        tmp_path: Path,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        shots_model: BaseMLModel,
        shots_experiment_config: ExperimentConfig,
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        mocker.patch("qxmt.datasets.DatasetBuilder.build", return_value=dataset)
        mocker.patch("qxmt.models.qkernels.KernelModelBuilder.build", return_value=shots_model)

        # initialization error check
        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db is None
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.run(config_source=shots_experiment_config, n_jobs=1)

        # run from config instance
        base_experiment.init()
        artifact, _ = base_experiment.run(config_source=shots_experiment_config, n_jobs=1)
        assert len(base_experiment.exp_db.runs) == 1  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_1/model.pkl").exists()
        assert base_experiment.experiment_dirc.joinpath("run_1/shots.h5").exists()
        assert base_experiment.experiment_dirc.joinpath("run_1/config.yaml").exists()
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)

        # run from config file
        experiment_config_file = tmp_path / "experiment_config.yaml"
        save_experiment_config_to_yaml(shots_experiment_config, experiment_config_file, delete_source_path=True)
        artifact, _ = base_experiment.run(config_source=experiment_config_file, add_results=True, n_jobs=1)
        assert len(base_experiment.exp_db.runs) == 2  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_2/model.pkl").exists()
        assert base_experiment.experiment_dirc.joinpath("run_2/shots.h5").exists()
        assert base_experiment.experiment_dirc.joinpath("run_2/config.yaml").exists()
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(artifact.dataset, Dataset)


class TestExperimentResults:
    def test_runs_to_dataframe(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_model: BaseMLModel
    ) -> None:
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.runs_to_dataframe()

        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        default_metrics_name = ["accuracy", "precision", "recall", "f1_score"]
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        df = base_experiment.runs_to_dataframe()

        assert Counter(df.columns) == Counter(["run_id", "accuracy", "precision", "recall", "f1_score"])
        assert len(df) == 2

    def test_runs_to_dataframe_with_validation(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_model: BaseMLModel
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
            model=state_vec_model,
            default_metrics_name=default_metrics_name,
            n_jobs=1,
        )
        base_experiment.run(
            model_type="qkernel",
            task_type="classification",
            dataset=dataset,
            model=state_vec_model,
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

    def test_save_experiment(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_model: BaseMLModel
    ) -> None:
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)

        base_experiment.init()
        base_experiment.run(
            model_type="qkernel", task_type="classification", dataset=dataset, model=state_vec_model, n_jobs=1
        )
        base_experiment.run(
            model_type="qkernel", task_type="classification", dataset=dataset, model=state_vec_model, n_jobs=1
        )
        base_experiment.save_experiment()

        assert (base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()


class TestExperimentReproduce:
    def test_reproduce(
        self, base_experiment: Experiment, create_random_dataset: Callable, state_vec_model: BaseMLModel
    ) -> None:
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.reproduce(run_id=1)

        base_experiment.init()
        dataset = create_random_dataset(data_num=100, feature_num=5, class_num=2)
        base_experiment.run(
            model_type="qkernel", task_type="classification", dataset=dataset, model=state_vec_model, n_jobs=1
        )

        # run_id=1 executed from dataset and model instance.
        # this run not exist config file.
        with pytest.raises(ReproductionError):
            base_experiment.reproduce(run_id=1)

        # run_id=2 is not exist in the experiment.
        with pytest.raises(ValueError):
            base_experiment.reproduce(run_id=2)

        # [TODO]: reproduce method not update experiment db and run_id.

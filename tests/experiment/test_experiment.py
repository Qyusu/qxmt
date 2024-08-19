import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from qxmt import Experiment
from qxmt.constants import DEFAULT_EXP_DB_FILE
from qxmt.exceptions import ExperimentNotInitializedError
from qxmt.models.base import BaseModel


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
    def set_dummy_experiment_data(self, experiment_dirc: Path) -> Path:
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
                    "commit_id": "commit_1",
                    "config_path": "config_1.yaml",
                    "evaluation": {
                        "accuracy": 0.65,
                        "precision": 0.69,
                        "recall": 0.65,
                        "f1_score": 0.67,
                    },
                },
                {
                    "run_id": 2,
                    "desc": "",
                    "execution_time": "2024-07-24 17:34:01.932990 JST+0900",
                    "commit_id": "commit_2",
                    "config_path": "",
                    "evaluation": {
                        "accuracy": 0.5,
                        "precision": 0.47,
                        "recall": 0.55,
                        "f1_score": 0.50,
                    },
                },
            ],
        }

        exp_json_path = Path(f"{dummy_experiment_dict['experiment_dirc']}/experiment.json")
        exp_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exp_json_path, "w") as json_file:
            json.dump(dummy_experiment_dict, json_file, indent=4)

        return exp_json_path

    def test_load_experiment(self, tmp_path: Path) -> None:
        exp_json_path = self.set_dummy_experiment_data(tmp_path)

        # Default pattern
        loaded_exp = Experiment(root_experiment_dirc=tmp_path).load_experiment(exp_json_path)
        assert loaded_exp.name == "load_exp"
        assert loaded_exp.desc == "load experiment"
        assert loaded_exp.root_experiment_dirc == tmp_path
        assert loaded_exp.experiment_dirc == tmp_path / "load_exp"
        assert loaded_exp.current_run_id == 2
        assert len(loaded_exp.exp_db.runs) == 2  # type: ignore

        # Update Setting Pattern
        updated_exp = Experiment(
            name="update_exp", desc="update experiment", root_experiment_dirc=tmp_path
        ).load_experiment(exp_json_path)
        assert updated_exp.name == "update_exp"
        assert updated_exp.desc == "update experiment"
        assert updated_exp.root_experiment_dirc == tmp_path
        assert updated_exp.experiment_dirc == tmp_path / "update_exp"
        assert updated_exp.current_run_id == 2
        assert len(updated_exp.exp_db.runs) == 2  # type: ignore


class TestExperimentRun:
    def test__run_setup(self, base_experiment: Experiment) -> None:
        base_experiment.init()
        assert base_experiment.current_run_id == 0
        assert not base_experiment.experiment_dirc.joinpath("run_1").exists()

        base_experiment._run_setup()
        assert base_experiment.current_run_id == 1
        assert base_experiment.experiment_dirc.joinpath("run_1").exists()

    def test_run_evaluation(self, base_experiment: Experiment) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        evaluation = base_experiment.run_evaluation(actual, predicted)
        acutal_result = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(evaluation) == 4
        for key, value in acutal_result.items():
            assert round(evaluation[key], 2) == value

    def test_run(
        self,
        base_experiment: Experiment,
        create_random_dataset: Callable,
        base_model: BaseModel,
    ) -> None:
        dataset = create_random_dataset(data_num=10, feature_num=5, class_num=2)

        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db is None
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.run(dataset=dataset, model=base_model)

        # run from dataset and model instance
        base_experiment.init()
        base_experiment.run(dataset=dataset, model=base_model)
        assert len(base_experiment.exp_db.runs) == 1  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_1/model.pkl").exists()

        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.run(dataset=dataset, model=base_model)
        assert len(base_experiment.exp_db.runs) == 3  # type: ignore

        # [TODO]: run from config file


class TestExperimentResults:
    def test_runs_to_dataframe(
        self, base_experiment: Experiment, create_random_dataset: Callable, base_model: BaseModel
    ) -> None:
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment._is_initialized()

        base_experiment.init()
        dataset = create_random_dataset(data_num=10, feature_num=5, class_num=2)
        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.run(dataset=dataset, model=base_model)
        df = base_experiment.runs_to_dataframe()

        assert len(df.columns) == 5
        assert len(df) == 2

    def test_save_experiment(
        self, base_experiment: Experiment, create_random_dataset: Callable, base_model: BaseModel
    ) -> None:
        dataset = create_random_dataset(data_num=10, feature_num=5, class_num=2)

        base_experiment.init()
        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.save_experiment()

        assert (base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()

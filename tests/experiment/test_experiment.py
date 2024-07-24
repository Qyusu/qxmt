import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from qk_manager import Experiment
from qk_manager.constants import DEFAULT_EXP_DB_FILE
from qk_manager.datasets.schema import Dataset
from qk_manager.exceptions import ExperimentNotInitializedError
from qk_manager.models.base_model import BaseModel


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
            "experiment_dirc": str(experiment_dirc / "load_exp"),
            "runs": [
                {
                    "run_id": 1,
                    "desc": "",
                    "execution_time": "2024-07-24 17:33:55.305025 JST+0900",
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
                    "evaluation": {
                        "accuracy": 0.5,
                        "precision": 0.47,
                        "recall": 0.55,
                        "f1_score": 0.50,
                    },
                },
            ],
        }
        exp_json_path = Path(dummy_experiment_dict["experiment_dirc"]) / "experiment.json"
        exp_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exp_json_path, "w") as json_file:
            json.dump(dummy_experiment_dict, json_file, indent=4)

        return exp_json_path

    def test_load_experiment(self, base_experiment: Experiment, tmp_path: Path) -> None:
        exp_json_path = self.set_dummy_experiment_data(tmp_path)
        base_experiment.load_experiment(exp_json_path)
        assert base_experiment.name == "load_exp"
        assert base_experiment.desc == "load experiment"
        assert base_experiment.experiment_dirc == tmp_path / "load_exp"
        assert base_experiment.current_run_id == 2
        assert len(base_experiment.exp_db.runs) == 2  # type: ignore


class TestExperimentRun:
    def test__run_setup(self, base_experiment: Experiment) -> None:
        assert base_experiment.current_run_id == 0
        assert not base_experiment.experiment_dirc.joinpath("run_1").exists()

        base_experiment._run_setup()
        assert base_experiment.current_run_id == 1
        assert base_experiment.experiment_dirc.joinpath("run_1").exists()

    def test__run_evaluation(self, base_experiment: Experiment) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        evaluation = base_experiment._run_evaluation(actual, predicted)
        acutal_result = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(evaluation) == 4
        for key, value in acutal_result.items():
            assert round(evaluation[key], 2) == value

    def test_run(self, base_experiment: Experiment, dataset: Dataset, base_model: BaseModel) -> None:
        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db is None
        with pytest.raises(ExperimentNotInitializedError):
            base_experiment.run(dataset=dataset, model=base_model)

        base_experiment.init()
        base_experiment.run(dataset=dataset, model=base_model)
        assert len(base_experiment.exp_db.runs) == 1  # type: ignore
        assert base_experiment.experiment_dirc.joinpath("run_1/model.pkl").exists()

        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.run(dataset=dataset, model=base_model)
        assert len(base_experiment.exp_db.runs) == 3  # type: ignore


class TestExperimentResults:
    # def test_runs_to_dataframe(self, base_experiment: Experiment) -> None:
    #     base_experiment.run()
    #     base_experiment.run()
    #     df = base_experiment.runs_to_dataframe()

    #     expected_df = pd.DataFrame(
    #         {
    #             "run_id": [1, 2],
    #             "accuracy": [],
    #             "precision": [],
    #             "recall": [],
    #             "f1_score": [],
    #         }
    #     )

    #     assert_frame_equal(df, expected_df)

    def test_save_experiment(self, base_experiment: Experiment, dataset: Dataset, base_model: BaseModel) -> None:
        base_experiment.init()
        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.run(dataset=dataset, model=base_model)
        base_experiment.save_experiment()

        assert (base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()

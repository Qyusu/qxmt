from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qk_manager import Experiment
from qk_manager.constants import DEFAULT_EXP_DB_FILE


class TestExperimentSettings:
    def test_default(self, tmp_path: Path) -> None:
        default_exp = Experiment(root_experiment_dirc=tmp_path)
        assert isinstance(default_exp.name, str)
        assert default_exp.desc == ""
        assert default_exp.current_run_id == 0
        assert default_exp.experiment_dirc.exists()

    def test_initialization(self, base_experiment: Experiment) -> None:
        assert base_experiment.name == "test_exp"
        assert base_experiment.desc == "test experiment"
        assert base_experiment.current_run_id == 0
        assert base_experiment.experiment_dirc.exists()


class TestExperimentInit:
    def test__init_db(self, tmp_path: Path) -> None:
        exp = Experiment(
            name="valid_test_exp",
            desc="valid test experiment",
            root_experiment_dirc=tmp_path,
        )
        exp_db = exp.exp_db

        assert exp_db.name == "valid_test_exp"
        assert exp_db.desc == "valid test experiment"
        assert exp_db.experiment_dirc == tmp_path / "valid_test_exp"


class TestExperimentRun:
    def test_run(self, base_experiment: Experiment) -> None:
        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db.runs == []

        base_experiment.run()
        assert base_experiment.current_run_id == 1
        assert len(base_experiment.exp_db.runs) == 1

        base_experiment.run()
        base_experiment.run()
        assert base_experiment.current_run_id == 3
        assert len(base_experiment.exp_db.runs) == 3

    def test__run_evaluation(self, base_experiment: Experiment) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        evaluation = base_experiment._run_evaluation(actual, predicted)
        acutal_result = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(evaluation) == 4
        for key, value in acutal_result.items():
            assert round(evaluation[key], 2) == value


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

    def test_save_experiment(self, base_experiment: Experiment) -> None:
        base_experiment.run()
        base_experiment.run()
        base_experiment.save_experiment()

        assert (base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()

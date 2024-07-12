from pathlib import Path

import pandas as pd

from qk_manager import Experiment
from qk_manager.constants import DEFAULT_EXP_DB_FILE, DEFAULT_RUN_DB_FILE


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

        assert exp_db.experiment_info.name == "valid_test_exp"
        assert exp_db.experiment_info.desc == "valid test experiment"
        assert exp_db.experiment_info.experiment_dirc == str(tmp_path / "valid_test_exp")


class TestExperimentRun:
    def test_run(self, base_experiment: Experiment) -> None:
        assert base_experiment.current_run_id == 0
        assert base_experiment.exp_db.run_info == []

        base_experiment.run()
        assert base_experiment.current_run_id == 1
        assert len(base_experiment.exp_db.run_info) == 1

        base_experiment.run()
        base_experiment.run()
        assert base_experiment.current_run_id == 3
        assert len(base_experiment.exp_db.run_info) == 3


class TestExperimentResults:
    def test_save_results(self, base_experiment: Experiment) -> None:
        base_experiment.run()
        base_experiment.run()
        base_experiment.save_results()

        assert (base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE).exists()
        assert (base_experiment.experiment_dirc / DEFAULT_RUN_DB_FILE).exists()

        exp_df = pd.read_csv(base_experiment.experiment_dirc / DEFAULT_EXP_DB_FILE)
        assert len(exp_df) == 1
        assert len(exp_df.columns) == 3

        run_df = pd.read_csv(base_experiment.experiment_dirc / DEFAULT_RUN_DB_FILE)
        assert len(run_df) == 2
        assert len(run_df.columns) == 1

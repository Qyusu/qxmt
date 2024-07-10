from pathlib import Path

import pytest

from qk_manager.experiment.experiment import Experiment


class TestExperimentSettings:
    @pytest.fixture(scope="function")
    def my_instance(self, tmp_path: Path) -> Experiment:
        return Experiment(
            name="test_exp",
            desc="test experiment",
            root_experiment_dirc=tmp_path,
        )

    def test_default(self, tmp_path: Path) -> None:
        default_exp = Experiment(root_experiment_dirc=tmp_path)
        assert isinstance(default_exp.name, str)
        assert default_exp.desc == ""
        assert default_exp.current_run_id == 0
        assert default_exp.experiment_dirc.exists()

    def test_initialization(self, my_instance: Experiment) -> None:
        assert my_instance.name == "test_exp"
        assert my_instance.desc == "test experiment"
        assert my_instance.current_run_id == 0
        assert my_instance.experiment_dirc.exists()

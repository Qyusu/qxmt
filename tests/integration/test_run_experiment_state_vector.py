from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import qxmt
from qxmt.configs import ExperimentConfig
from qxmt.datasets import Dataset
from qxmt.experiment import RunArtifact, RunRecord
from qxmt.models import BaseMLModel


class TestRunExperiment:
    def test_run_experiment_by_state_vector_simulator_from_config_file(self, tmp_path: Path) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_by_state_vector_simulator_from_config_file",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the state vector simulator from config file.
            """,
            auto_gen_mode=False,
        ).init()

        # check to create the experiment directory
        assert (tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file").exists()

        # run by config file
        config_path = "tests/integration/configs/state_vector_simulator.yaml"
        artifact, result = experiment.run(config_source=config_path)

        # check return values
        assert isinstance(artifact, RunArtifact)
        assert artifact.run_id == 1
        assert isinstance(artifact.dataset, Dataset)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(result, RunRecord)
        assert len(experiment.exp_db.runs) == 1  # type: ignore

        # check saved artifacts
        assert (
            tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file/run_1/config.yaml"
        ).exists()
        assert (
            tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file/run_1/model.pkl"
        ).exists()
        assert not (
            tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file/run_1/shots.h5"
        ).exists()

        # check update run id
        artifact, result = experiment.run(config_source=config_path)
        assert artifact.run_id == 2
        assert len(experiment.exp_db.runs) == 2  # type: ignore

        # check saved artifacts
        assert (
            tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file/run_2/config.yaml"
        ).exists()
        assert (
            tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file/run_2/model.pkl"
        ).exists()
        assert not (
            tmp_path / "experiments/integration_test_by_state_vector_simulator_from_config_file/run_2/shots.h5"
        ).exists()

    @pytest.mark.parametrize(
        "device_name, kernel_name, expected_df",
        [
            pytest.param(
                "default.qubit",
                "FidelityKernel",
                pd.DataFrame(
                    {
                        "run_id": [1],
                        "accuracy": [0.45],
                        "precision": [0.57],
                        "recall": [0.36],
                        "f1_score": [0.37],
                    }
                ),
                id="default.qubit and FidelityKernel",
            ),
            pytest.param(
                "lightling.qubit",
                "FidelityKernel",
                pd.DataFrame(
                    {
                        "run_id": [1],
                        "accuracy": [0.45],
                        "precision": [0.57],
                        "recall": [0.36],
                        "f1_score": [0.37],
                    }
                ),
                id="lightling.qubit and FidelityKernel",
            ),
        ],
    )
    def test_run_experiment_by_state_vector_simulator_from_config_instance(
        self, device_name: str, kernel_name: str, expected_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_by_state_vector_simulator_from_config_instance",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the state vector simulator from config instance.
            """,
            auto_gen_mode=False,
        ).init()

        # update config
        base_config_path = "tests/integration/configs/state_vector_simulator.yaml"
        base_config = ExperimentConfig(path=base_config_path)
        config = base_config.model_copy(update={"device.name": device_name, "kernel.implement_name": kernel_name})
        _, _ = experiment.run(config_source=config)

        # get result dataframe
        # compare up to 2 decimal places
        result_df = experiment.runs_to_dataframe().round(2)
        assert_frame_equal(result_df, expected_df.round(2))

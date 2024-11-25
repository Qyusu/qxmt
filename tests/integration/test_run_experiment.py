import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import qxmt
from qxmt.configs import ExperimentConfig
from qxmt.datasets import Dataset
from qxmt.experiment import RunArtifact, RunRecord
from qxmt.models import BaseMLModel


class TestRunExperiment:
    def test_run_experiment_by_state_vector_simulator(self, tmp_path: Path) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_by_state_vector_simulator",
            root_experiment_dirc=tmp_path / "experiments",
            desc="This is an integration test for running an experiment by the state vector simulator.",
            auto_gen_mode=False,
        ).init()

        # check to create the experiment directory
        assert (tmp_path / "experiments/integration_test_by_state_vector_simulator").exists()

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
        assert (tmp_path / "experiments/integration_test_by_state_vector_simulator/run_1/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_by_state_vector_simulator/run_1/model.pkl").exists()
        assert not (tmp_path / "experiments/integration_test_by_state_vector_simulator/run_1/shots.h5").exists()

        # update config
        adhoc_config = ExperimentConfig(path=config_path)
        adhoc_config.model.params.update({"C": 0.5, "gamma": 0.01})
        artifact, result = experiment.run(config_source=adhoc_config)

        # check return values
        assert artifact.run_id == 2
        assert len(experiment.exp_db.runs) == 2  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_by_state_vector_simulator/run_2/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_by_state_vector_simulator/run_2/model.pkl").exists()
        assert not (tmp_path / "experiments/integration_test_by_state_vector_simulator/run_2/shots.h5").exists()

        # get result dataframe
        # compare up to 2 decimal places
        result_df = experiment.runs_to_dataframe().round(2)
        expected_df = pd.DataFrame(
            {
                "run_id": [1, 2],
                "accuracy": [0.45, 0.35],
                "precision": [0.57, 0.21],
                "recall": [0.36, 0.27],
                "f1_score": [0.37, 0.23],
            }
        ).round(2)
        assert_frame_equal(result_df, expected_df)

    def test_run_experiment_by_sampling_simulator(self, tmp_path: Path) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_by_sampling_simulator",
            root_experiment_dirc=tmp_path / "experiments",
            desc="This is an integration test for running an experiment by the sampling simulator.",
            auto_gen_mode=False,
        ).init()

        # check to create the experiment directory
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator").exists()

        # run by config file
        config_path = "tests/integration/configs/sampling_simulator.yaml"
        artifact, result = experiment.run(config_source=config_path)

        # check return values
        assert isinstance(artifact, RunArtifact)
        assert artifact.run_id == 1
        assert isinstance(artifact.dataset, Dataset)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(result, RunRecord)
        assert len(experiment.exp_db.runs) == 1  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator/run_1/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator/run_1/model.pkl").exists()
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator/run_1/shots.h5").exists()

        # update config
        adhoc_config = ExperimentConfig(path=config_path)
        adhoc_config.model.params.update({"C": 0.5, "gamma": 0.01})
        artifact, result = experiment.run(config_source=adhoc_config)

        # check return values
        assert artifact.run_id == 2
        assert len(experiment.exp_db.runs) == 2  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator/run_2/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator/run_2/model.pkl").exists()
        assert (tmp_path / "experiments/integration_test_by_sampling_simulator/run_2/shots.h5").exists()

        # get result dataframe
        # compare up to 2 decimal places
        result_df = experiment.runs_to_dataframe().round(2)
        print("*" * 80)
        print(f"Result DataFrame: {result_df}")
        if sys.version_info[:2] == (3, 10):
            expected_df = pd.DataFrame(
                {
                    "run_id": [1, 2],
                    "accuracy": [0.40, 0.35],
                    "precision": [0.55, 0.20],
                    "recall": [0.33, 0.27],
                    "f1_score": [0.35, 0.23],
                }
            ).round(2)
        elif sys.version_info[:2] == (3, 11):
            expected_df = pd.DataFrame(
                {
                    "run_id": [1, 2],
                    "accuracy": [0.40, 0.35],
                    "precision": [0.55, 0.20],
                    "recall": [0.33, 0.27],
                    "f1_score": [0.35, 0.23],
                }
            ).round(2)
        else:
            raise ValueError("Unsupported Python version")

        assert_frame_equal(result_df, expected_df, check_exact=False, atol=1e-1)

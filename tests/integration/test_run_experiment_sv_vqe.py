import platform
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import qxmt
from qxmt.configs import ExperimentConfig
from qxmt.experiment import RunArtifact, RunRecord
from qxmt.models.vqe import BaseVQE


class TestRunExperimentStateVectorVQE:
    def test_run_experiment_from_config_file(self, tmp_path: Path) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_sv_vqe",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the state vector simulator from config file.
            """,
            auto_gen_mode=False,
        ).init()

        # check to create the experiment directory
        assert (tmp_path / "experiments/integration_test_sv_vqe").exists()

        # run by config file
        config_path = "tests/integration/configs/simulator_sv_vqe.yaml"
        artifact, result = experiment.run(config_source=config_path)

        # check return values
        assert isinstance(artifact, RunArtifact)
        assert artifact.run_id == 1
        assert artifact.dataset is None
        assert isinstance(artifact.model, BaseVQE)
        assert isinstance(result, RunRecord)
        assert len(experiment.exp_db.runs) == 1  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_sv_vqe/run_1/config.yaml").exists()

        # check update run id
        artifact, result = experiment.run(config_source=config_path)
        assert artifact.run_id == 2
        assert len(experiment.exp_db.runs) == 2  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_sv_vqe/run_2/config.yaml").exists()

    @pytest.mark.parametrize(
        "device_name",
        [
            pytest.param(
                "default.qubit",
                id="default.qubit",
            ),
            pytest.param(
                "lightning.qubit",
                id="lightning.qubit",
            ),
            pytest.param(
                "qulacs.simulator",
                id="qulacs.simulator",
            ),
        ],
    )
    def test_run_experiment_by_state_vector_simulator_from_config_instance(
        self, device_name: str, tmp_path: Path
    ) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_sv_vqe",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the state vector simulator from config instance.
            """,
            auto_gen_mode=False,
        ).init()

        # update config
        base_config_path = "tests/integration/configs/simulator_sv_vqe.yaml"
        base_config = ExperimentConfig(path=base_config_path)
        updated_device = base_config.device.model_copy(update={"device_name": device_name})
        config = base_config.model_copy(update={"device": updated_device})

        _, _ = experiment.run(config_source=config)

        # get result dataframe
        # compare up to 2 decimal places
        result_df = experiment.runs_to_dataframe(include_validation=True).round(2)
        expected_df = pd.DataFrame(
            {
                "run_id": [1],
                "final_cost": [-1.14],
                "hf_energy": [-1.12],
            }
        ).round(2)

        assert_frame_equal(result_df, expected_df)

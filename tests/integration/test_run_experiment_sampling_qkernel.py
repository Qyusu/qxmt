from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import qxmt
from qxmt.configs import ExperimentConfig
from qxmt.datasets import Dataset
from qxmt.experiment import RunArtifact, RunRecord
from qxmt.models.qkernels import BaseMLModel


class TestRunExperimentSamplingQKernel:
    def test_run_experiment_by_sampling_simulator_from_config_file(self, tmp_path: Path) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_sampling_qkernel",
            root_experiment_dirc=tmp_path / "experiments",
            desc="This is an integration test for running an experiment by the sampling simulator from config file.",
            auto_gen_mode=False,
        ).init()

        # check to create the experiment directory
        assert (tmp_path / "experiments/integration_test_sampling_qkernel").exists()

        # run by config file
        config_path = "tests/integration/configs/simulator_sampling_qkernel.yaml"
        artifact, result = experiment.run(config_source=config_path)

        # check return values
        assert isinstance(artifact, RunArtifact)
        assert artifact.run_id == 1
        assert isinstance(artifact.dataset, Dataset)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(result, RunRecord)
        assert len(experiment.exp_db.runs) == 1  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_sampling_qkernel/run_1/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_sampling_qkernel/run_1/model.pkl").exists()
        assert (tmp_path / "experiments/integration_test_sampling_qkernel/run_1/shots.h5").exists()

        # check update run id
        artifact, result = experiment.run(config_source=config_path)
        assert artifact.run_id == 2
        assert len(experiment.exp_db.runs) == 2  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_sampling_qkernel/run_2/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_sampling_qkernel/run_2/model.pkl").exists()
        assert (tmp_path / "experiments/integration_test_sampling_qkernel/run_2/shots.h5").exists()

    @pytest.mark.parametrize(
        "device_name, kernel_name",
        [
            pytest.param(
                "default.qubit",
                "FidelityKernel",
                id="default.qubit and FidelityKernel",
            ),
            pytest.param(
                "lightning.qubit",
                "FidelityKernel",
                id="lightling.qubit and FidelityKernel",
            ),
            pytest.param(
                "qulacs.simulator",
                "FidelityKernel",
                id="qulacs.simulator and FidelityKernel",
            ),
        ],
    )
    def test_run_experiment_by_sampling_simulator_from_config_instance(
        self,
        device_name: str,
        kernel_name: str,
        tmp_path: Path,
    ) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_sampling_qkernel",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the sampling simulator from config instance.
            """,
            auto_gen_mode=False,
        ).init()

        # update config
        base_config_path = "tests/integration/configs/simulator_sampling_qkernel.yaml"
        base_config = ExperimentConfig(path=base_config_path)
        updated_device = base_config.device.model_copy(update={"device_name": device_name})
        updated_kernel = (
            base_config.kernel.model_copy(update={"implement_name": kernel_name}) if base_config.kernel else None
        )
        config = base_config.model_copy(update={"device": updated_device, "kernel": updated_kernel})

        _, _ = experiment.run(config_source=config)

        # get result dataframe, and compare up to 2 decimal places
        result_df = experiment.runs_to_dataframe().round(2)
        for _, row in result_df.iterrows():
            assert row.accuracy >= 0.1
            assert row.precision >= 0.1
            assert row.recall >= 0.1
            assert row.f1_score >= 0.1

        # [TODO]: check the result is correct
        # expected_df = pd.DataFrame(
        #     {
        #         "run_id": [1],
        #         "accuracy": [0.55],
        #         "precision": [0.50],
        #         "recall": [0.52],
        #         "f1_score": [0.50],
        #     }
        # ).round(2)

        # assert_frame_equal(result_df, expected_df, atol=0.1, check_exact=False)

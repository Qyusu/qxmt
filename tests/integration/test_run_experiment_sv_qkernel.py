import platform
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import qxmt
from qxmt.configs import ExperimentConfig
from qxmt.datasets import Dataset
from qxmt.experiment import RunArtifact, RunRecord
from qxmt.models.qkernels import BaseMLModel

EXPECTED_RESULT_BY_PLATFORM = {
    ("Linux", "x86_64"): {
        "run_id": [1],
        "accuracy": [0.60],
        "precision": [0.60],
        "recall": [0.61],
        "f1_score": [0.60],
        "accuracy_validation": [0.67],
        "precision_validation": [0.67],
        "recall_validation": [0.69],
        "f1_score_validation": [0.67],
    },
    ("Darwin", "arm64"): {
        "run_id": [1],
        "accuracy": [0.43],
        "precision": [0.54],
        "recall": [0.47],
        "f1_score": [0.45],
        "accuracy_validation": [0.57],
        "precision_validation": [0.58],
        "recall_validation": [0.58],
        "f1_score_validation": [0.54],
    },
}


class TestRunExperimentStateVectorQKernel:
    @pytest.mark.parametrize(
        "config_path",
        [
            pytest.param(
                "tests/integration/configs/simulator_sv_qkernel_pennylane.yaml",
                id="pennylane config file",
            ),
            pytest.param(
                "tests/integration/configs/simulator_sv_qkernel_qulacs.yaml",
                id="qulacs config file",
            ),
        ],
    )
    def test_run_experiment_from_config_file(self, config_path: str, tmp_path: Path) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_sv_qkernel",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the state vector simulator from config file.
            """,
            auto_gen_mode=False,
        ).init()

        # check to create the experiment directory
        assert (tmp_path / "experiments/integration_test_sv_qkernel").exists()

        # run by config file
        artifact, result = experiment.run(config_source=config_path)

        # check return values
        assert isinstance(artifact, RunArtifact)
        assert artifact.run_id == 1
        assert isinstance(artifact.dataset, Dataset)
        assert isinstance(artifact.model, BaseMLModel)
        assert isinstance(result, RunRecord)
        assert len(experiment.exp_db.runs) == 1  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_sv_qkernel/run_1/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_sv_qkernel/run_1/model.pkl").exists()
        assert not (tmp_path / "experiments/integration_test_sv_qkernel/run_1/shots.h5").exists()

        # check update run id
        artifact, result = experiment.run(config_source=config_path)
        assert artifact.run_id == 2
        assert len(experiment.exp_db.runs) == 2  # type: ignore

        # check saved artifacts
        assert (tmp_path / "experiments/integration_test_sv_qkernel/run_2/config.yaml").exists()
        assert (tmp_path / "experiments/integration_test_sv_qkernel/run_2/model.pkl").exists()
        assert not (tmp_path / "experiments/integration_test_sv_qkernel/run_2/shots.h5").exists()

    @pytest.mark.parametrize(
        "config_path, device_name, kernel_name",
        [
            pytest.param(
                "tests/integration/configs/simulator_sv_qkernel_pennylane.yaml",
                "default.qubit",
                "FidelityKernel",
                id="PennyLane default.qubit and FidelityKernel",
            ),
            pytest.param(
                "tests/integration/configs/simulator_sv_qkernel_pennylane.yaml",
                "lightning.qubit",
                "FidelityKernel",
                id="PennyLane lightning.qubit and FidelityKernel",
            ),
            pytest.param(
                "tests/integration/configs/simulator_sv_qkernel_pennylane.yaml",
                "qulacs.simulator",
                "FidelityKernel",
                id="PennyLane qulacs.simulator and FidelityKernel",
            ),
            pytest.param(
                "tests/integration/configs/simulator_sv_qkernel_qulacs.yaml",
                "cpu.simulator",
                "FidelityKernel",
                id="Qulacs cpu.simulator and FidelityKernel",
            ),
        ],
    )
    def test_run_experiment_by_pennylane_state_vector_simulator_from_config_instance(
        self, config_path: str, device_name: str, kernel_name: str, tmp_path: Path
    ) -> None:
        experiment = qxmt.Experiment(
            name="integration_test_sv_qkernel",
            root_experiment_dirc=tmp_path / "experiments",
            desc="""
            This is an integration test for running an experiment by the state vector simulator from config instance.
            """,
            auto_gen_mode=False,
        ).init()

        # update config
        base_config = ExperimentConfig(path=config_path)
        updated_device = base_config.device.model_copy(update={"device_name": device_name})
        updated_kernel = (
            base_config.kernel.model_copy(update={"implement_name": kernel_name}) if base_config.kernel else None
        )
        config = base_config.model_copy(update={"device": updated_device, "kernel": updated_kernel})

        _, _ = experiment.run(config_source=config)

        # get result dataframe
        # compare up to 2 decimal places
        result_df = experiment.runs_to_dataframe(include_validation=True).round(2)
        platform_key = (platform.system(), platform.machine())
        if platform_key not in EXPECTED_RESULT_BY_PLATFORM:
            raise ValueError(f"Unsupported platform: {platform_key}")

        expected_df = pd.DataFrame(EXPECTED_RESULT_BY_PLATFORM[platform_key]).round(2)

        try:
            assert_frame_equal(result_df, expected_df)
        except AssertionError as exc:
            raise AssertionError(
                "Result metrics did not match expected values.\n"
                f"platform={platform_key}\n"
                f"actual={result_df.to_dict(orient='list')}\n"
                f"expected={expected_df.to_dict(orient='list')}"
            ) from exc

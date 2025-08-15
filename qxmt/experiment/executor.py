from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

from qxmt.constants import (
    DEFAULT_EXP_CONFIG_FILE,
    DEFAULT_MODEL_NAME,
    DEFAULT_SHOT_RESULTS_NAME,
    TZ,
)
from qxmt.datasets import Dataset, DatasetBuilder
from qxmt.experiment.evaluation_factory import (
    QKERNEL_MODEL_TYPE_NAME,
    VQE_MODEL_TYPE_NAME,
    EvaluationFactory,
)
from qxmt.experiment.schema import (
    Evaluations,
    RemoteMachine,
    RunArtifact,
    RunRecord,
    RunTime,
    VQEEvaluations,
    VQERunTime,
)
from qxmt.logger import set_default_logger
from qxmt.models import ModelBuilder
from qxmt.models.qkernels import BaseKernelModel, BaseMLModel
from qxmt.models.vqe import BaseVQE, VQEModelBuilder

if TYPE_CHECKING:  # pragma: no cover
    # avoid circular import
    from qxmt.configs import ExperimentConfig
    from qxmt.experiment.experiment import Experiment

LOGGER = set_default_logger(__name__)


class RunExecutorBase(ABC):
    """Abstract base class for experiment run executors.

    This class provides common functionality and attributes for concrete executor
    implementations. It serves as a base class for both QKernel and VQE executors.

    Attributes:
        exp (Experiment): Reference to the experiment instance.
        logger: Logger instance for logging operations.
        eval_factory (EvaluationFactory): Factory for creating evaluation instances.
    """

    def __init__(self, experiment: "Experiment") -> None:
        """Initialize the base executor.

        Args:
            experiment (Experiment): The experiment instance to use for execution.
        """
        self.exp = experiment  # hold reference for common helpers/loggers
        self.logger = experiment.logger
        self.eval_factory = EvaluationFactory

    @abstractmethod
    def run_from_config(self, *args: Any, **kwargs: Any) -> tuple[RunArtifact, RunRecord]:
        """Run a model using configuration settings."""
        pass

    @abstractmethod
    def run_from_instance(self, *args: Any, **kwargs: Any) -> tuple[RunArtifact, RunRecord]:
        """Run a model using a pre-built instance."""
        pass


class QKernelExecutor(RunExecutorBase):
    """Executor for QKernel models (classification / regression).

    This class handles the execution of quantum kernel-based machine learning
    models, including both classification and regression tasks.
    """

    def run_from_config(
        self,
        *,
        config: "ExperimentConfig",
        commit_id: str,
        run_dirc: Path,
        n_jobs: int,
        show_progress: bool,
        repo_path: Optional[str],
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Run a QKernel model using configuration settings.

        Args:
            config (ExperimentConfig): Configuration settings for the experiment.
            commit_id (str): Git commit ID for the current run.
            run_dirc (Path): Directory to store run artifacts.
            n_jobs (int): Number of parallel jobs to run.
            show_progress (bool): Whether to display progress bars.
            repo_path (Optional[str]): Path to the repository.
            add_results (bool, optional): Whether to save results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: Tuple containing the run artifact and record.

        Note:
            This method builds the dataset and model from the configuration before
            executing the run.
        """
        # Build dataset and model from config
        dataset = DatasetBuilder(config=config).build()
        model = ModelBuilder(config=config, n_jobs=n_jobs, show_progress=show_progress).build()

        save_shots_path = run_dirc / DEFAULT_SHOT_RESULTS_NAME if add_results else None
        save_model_path = run_dirc / DEFAULT_MODEL_NAME

        return self.run_from_instance(
            task_type=config.global_settings.task_type,
            dataset=dataset,
            model=cast(BaseMLModel, model),
            save_shots_path=save_shots_path,
            save_model_path=save_model_path,
            default_metrics_name=config.evaluation.default_metrics,
            custom_metrics=config.evaluation.custom_metrics,
            desc=config.description,
            commit_id=commit_id,
            config_file_name=Path(DEFAULT_EXP_CONFIG_FILE),
            repo_path=repo_path,
            add_results=add_results,
        )

    def run_from_instance(
        self,
        *,
        task_type: Optional[str],
        dataset: Dataset,
        model: BaseMLModel,
        save_shots_path: Optional[Path],
        save_model_path: Path,
        default_metrics_name: Optional[list[str]],
        custom_metrics: Optional[list[dict[str, Any]]],
        desc: str,
        commit_id: str,
        config_file_name: Path,
        repo_path: Optional[str],
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Run a QKernel model using pre-built instances.

        This method executes the model training, validation, and testing phases,
        and records the results.

        Args:
            task_type (Optional[str]): Type of the task (classification/regression).
            dataset (Dataset): Dataset to use for training and evaluation.
            model (BaseMLModel): Pre-built model instance.
            save_shots_path (Optional[Path]): Path to save shot results.
            save_model_path (Path): Path to save the trained model.
            default_metrics_name (Optional[list[str]]): List of default metrics to use.
            custom_metrics (Optional[list[dict[str, Any]]]): List of custom metrics.
            desc (str): Description of the run.
            commit_id (str): Git commit ID.
            config_file_name (Path): Name of the configuration file.
            repo_path (Optional[str]): Path to the repository.
            add_results (bool, optional): Whether to save results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: Tuple containing the run artifact and record.
        """
        # Training phase
        train_start_dt = datetime.now(TZ)
        model.fit(X=dataset.X_train, y=dataset.y_train, save_shots_path=save_shots_path)
        train_end_dt = datetime.now(TZ)
        train_seconds = (train_end_dt - train_start_dt).total_seconds()

        # Validation phase (optional)
        if (dataset.X_val is not None) and (dataset.y_val is not None):
            val_start_dt = datetime.now(TZ)
            y_val_pred = model.predict(dataset.X_val, bar_label="Validation")
            val_end_dt = datetime.now(TZ)
            val_seconds = (val_end_dt - val_start_dt).total_seconds()
            val_eval = self.eval_factory.evaluate(
                model_type=QKERNEL_MODEL_TYPE_NAME,
                task_type=task_type,
                params={"actual": dataset.y_val, "predicted": y_val_pred},
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        else:
            val_seconds = None
            val_eval = None

        # Test phase
        test_start_dt = datetime.now(TZ)
        y_test_pred = model.predict(dataset.X_test, bar_label="Test")
        test_end_dt = datetime.now(TZ)
        test_seconds = (test_end_dt - test_start_dt).total_seconds()
        test_eval = self.eval_factory.evaluate(
            model_type=QKERNEL_MODEL_TYPE_NAME,
            task_type=task_type,
            params={"actual": dataset.y_test, "predicted": y_test_pred},
            default_metrics_name=default_metrics_name,
            custom_metrics=custom_metrics,
        )

        # Remote machine log if device is remote
        device = cast(BaseKernelModel, model).kernel.device
        if device.is_remote():
            train_job_ids = device.get_job_ids(created_after=train_start_dt, created_before=train_end_dt)
            val_job_ids = device.get_job_ids(created_after=val_start_dt, created_before=val_end_dt) if val_eval else []
            test_job_ids = device.get_job_ids(created_after=test_start_dt, created_before=test_end_dt)
            remote_machine = RemoteMachine(
                provider=device.get_provider(),
                backend=device.get_backend_name(),
                job_ids=train_job_ids + val_job_ids + test_job_ids,
            )
        else:
            remote_machine = None

        if add_results:
            model.save(save_model_path)

        artifact = RunArtifact(run_id=self.exp.current_run_id, dataset=dataset, model=model)
        record = RunRecord(
            run_id=self.exp.current_run_id,
            desc=self.exp._get_auto_description(desc, repo_path),
            remote_machine=remote_machine,
            commit_id=commit_id,
            config_file_name=config_file_name,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            runtime=RunTime(
                train_seconds=train_seconds,
                validation_seconds=val_seconds,
                test_seconds=test_seconds,
            ),
            evaluations=Evaluations(validation=val_eval, test=test_eval),
        )
        return artifact, record


class VQEExecutor(RunExecutorBase):
    """Executor for Variational Quantum Eigensolver models.

    This class handles the execution of VQE models, which are used for finding
    the ground state energy of quantum systems.
    """

    def run_from_config(
        self,
        *,
        config: "ExperimentConfig",
        commit_id: str,
        run_dirc: Path,
        n_jobs: int,
        show_progress: bool,
        repo_path: Optional[str],
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Run a VQE model using configuration settings.

        Args:
            config (ExperimentConfig): Configuration settings for the experiment.
            commit_id (str): Git commit ID for the current run.
            run_dirc (Path): Directory to store run artifacts.
            n_jobs (int): Number of parallel jobs to run.
            show_progress (bool): Whether to display progress bars.
            repo_path (Optional[str]): Path to the repository.
            add_results (bool, optional): Whether to save results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: Tuple containing the run artifact and record.
        """
        model = VQEModelBuilder(config=config, n_jobs=n_jobs).build()
        # [TODO]: implement save shots logic
        save_shots_path = run_dirc / DEFAULT_SHOT_RESULTS_NAME if add_results else None
        return self.run_from_instance(
            model=model,
            save_shots_path=save_shots_path,
            default_metrics_name=config.evaluation.default_metrics,
            custom_metrics=config.evaluation.custom_metrics,
            desc=config.description,
            commit_id=commit_id,
            config_file_name=Path(DEFAULT_EXP_CONFIG_FILE),
            repo_path=repo_path,
            add_results=add_results,
        )

    def run_from_instance(
        self,
        *,
        model: BaseVQE,
        save_shots_path: Optional[Path],
        default_metrics_name: Optional[list[str]],
        custom_metrics: Optional[list[dict[str, Any]]],
        desc: str,
        commit_id: str,
        config_file_name: Path,
        repo_path: Optional[str],
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Run a VQE model using a pre-built instance.

        This method executes the VQE optimization process and records the results.

        Args:
            model (BaseVQE): Pre-built VQE model instance.
            save_shots_path (Optional[Path]): Path to save shot results.
            default_metrics_name (Optional[list[str]]): List of default metrics to use.
            custom_metrics (Optional[list[dict[str, Any]]]): List of custom metrics.
            desc (str): Description of the run.
            commit_id (str): Git commit ID.
            config_file_name (Path): Name of the configuration file.
            repo_path (Optional[str]): Path to the repository.
            add_results (bool, optional): Whether to save results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: Tuple containing the run artifact and record.

        Raises:
            NotImplementedError: If remote machine execution is attempted.
        """
        # Optimise the VQE model
        optimise_start = datetime.now(TZ)
        model.optimize(init_params=None)
        optimise_end = datetime.now(TZ)
        optimise_seconds = (optimise_end - optimise_start).total_seconds()

        evaluations = self.eval_factory.evaluate(
            model_type=VQE_MODEL_TYPE_NAME,
            task_type=None,
            params={"cost_history": model.cost_history, "hamiltonian": model.hamiltonian},
            default_metrics_name=default_metrics_name,
            custom_metrics=custom_metrics,
        )

        if model.device.is_remote():
            raise NotImplementedError("Remote machine is not supported for VQE.")
        remote_machine = None

        artifact = RunArtifact(run_id=self.exp.current_run_id, dataset=None, model=model)

        record = RunRecord(
            run_id=self.exp.current_run_id,
            desc=self.exp._get_auto_description(desc, repo_path),
            remote_machine=remote_machine,
            commit_id=commit_id,
            config_file_name=config_file_name,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            runtime=VQERunTime(
                optimize_seconds=optimise_seconds,
                circuit_depth=model.get_circuit_depth(),
                n_parameters=model.ansatz.n_params,
            ),
            evaluations=VQEEvaluations(optimized=evaluations),
        )
        return artifact, record

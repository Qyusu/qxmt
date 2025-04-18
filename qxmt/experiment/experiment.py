import json
import os
import shutil
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd

from qxmt.configs import ExperimentConfig
from qxmt.constants import (
    DEFAULT_EXP_CONFIG_FILE,
    DEFAULT_EXP_DB_FILE,
    DEFAULT_EXP_DIRC,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_JOBS,
    DEFAULT_SHOT_RESULTS_NAME,
    LLM_MODEL_PATH,
    TZ,
)
from qxmt.datasets import Dataset, DatasetBuilder
from qxmt.evaluation import (
    ClassificationEvaluation,
    RegressionEvaluation,
    VQEEvaluation,
)
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    ExperimentSettingError,
    InvalidFileExtensionError,
    JsonEncodingError,
    ReproductionError,
)
from qxmt.experiment.schema import (
    Evaluations,
    ExperimentDB,
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
from qxmt.models.vqe import BaseVQE, VQEBuilder
from qxmt.utils import (
    get_commit_id,
    get_git_add_code,
    get_git_rm_code,
    is_git_available,
    save_experiment_config_to_yaml,
)

USE_LLM = os.getenv("USE_LLM", "FALSE").lower() == "true"
if USE_LLM:
    from qxmt.generators import DescriptionGenerator

IS_GIT_AVAILABLE = is_git_available()

LOGGER = set_default_logger(__name__)

QKERNEL_MODEL_TYPE_NAME: str = "qkernel"
VQE_MODEL_TYPE_NAME: str = "vqe"


class Experiment:
    """Experiment class for managing quantum machine learning experiments and their run data.

    The Experiment class provides a comprehensive framework for:
    - Initializing and managing quantum machine learning experiments
    - Running experiments with different configurations
    - Tracking and storing experiment results
    - Reproducing previous experiments
    - Managing experiment artifacts and metadata

    All experiment data is stored in an ExperimentDB instance, which is saved as a JSON file
    in the experiment directory (root_experiment_dirc/experiments/your_exp_name/experiment.json).

    The Experiment class supports two main modes of operation:

    1. **Config-based Mode**:
    This mode uses a YAML configuration file to define experiment settings.
    It provides full tracking of experiment settings, results, and enables model reproduction.
    This is the recommended approach for production use.

    2. **Instance-based Mode**:
    This mode directly accepts dataset and model instances.
    While simpler to use, it does not track experiment settings and is primarily intended
    for quick testing, debugging, or ad-hoc experiments.

    The class supports both quantum kernel models (QKernel) and Variational Quantum Eigensolver (VQE) models,
    with appropriate evaluation metrics for each type.

    Examples:
        >>> import qxmt
        >>> # Initialize experiment with auto-generated description
        >>> exp = qxmt.Experiment(
        ...     name="my_qsvm_algorithm",
        ...     desc='''This is an experiment for testing a new QSVM algorithm.
        ...     The experiment evaluates performance across multiple datasets.''',
        ...     auto_gen_mode=True,
        ... ).init()

        >>> # Run experiment using config file
        >>> config_path = "../configs/template.yaml"
        >>> artifact, result = exp.run(config_source=config_path)

        >>> # View experiment results
        >>> exp.runs_to_dataframe()
            run_id	accuracy	precision	recall	f1_score
        0	     1	    0.45	     0.53	  0.66	    0.59
    """

    def __init__(
        self,
        name: Optional[str] = None,
        desc: Optional[str] = None,
        auto_gen_mode: bool = USE_LLM,
        root_experiment_dirc: str | Path = DEFAULT_EXP_DIRC,
        llm_model_path: str = LLM_MODEL_PATH,
        logger: Logger = LOGGER,
    ) -> None:
        """Initialize the Experiment class with experiment settings and configuration.

        This method sets up the basic configuration for the experiment, including:
        - Experiment name and description
        - Auto-generation mode for descriptions
        - Experiment directory structure
        - LLM model path (if using auto-generation)
        - Logger configuration

        Args:
            name (Optional[str], optional):
                Name of the experiment. If None, a default name will be generated
                using the current timestamp. Defaults to None.
            desc (Optional[str], optional):
                Description of the experiment. Used for documentation and search purposes.
                If None and auto_gen_mode is True, a description will be generated.
                Defaults to None.
            auto_gen_mode (bool, optional):
                Whether to use the DescriptionGenerator for automatic description generation.
                Requires the "USE_LLM" environment variable to be set to True.
                Defaults to USE_LLM.
            root_experiment_dirc (str | Path, optional):
                Root directory where experiment data will be stored.
                Defaults to DEFAULT_EXP_DIRC.
            llm_model_path (str, optional):
                Path to the LLM model used for description generation.
                Defaults to LLM_MODEL_PATH.
            logger (Logger, optional):
                Logger instance for handling warning and error messages.
                Defaults to LOGGER.

        Note:
            If auto_gen_mode is True but USE_LLM is False, a warning will be logged
            and auto_gen_mode will be automatically set to False.
        """
        self.name: Optional[str] = name
        self.desc: Optional[str] = desc
        self.auto_gen_mode: bool = auto_gen_mode
        self.current_run_id: int = 0
        self.root_experiment_dirc: Path = Path(root_experiment_dirc)
        self.experiment_dirc: Path
        self.exp_db: Optional[ExperimentDB] = None
        self.logger: Logger = logger

        if (not USE_LLM) and (self.auto_gen_mode):
            self.logger.warning(
                'Global variable "USE_LLM" is set to False. '
                'DescriptionGenerator is not available. Set "USE_LLM" to True to use DescriptionGenerator.'
            )
            self.auto_gen_mode = False
        elif self.auto_gen_mode:
            self.desc_generator = DescriptionGenerator(llm_model_path)

    @staticmethod
    def _generate_default_name() -> str:
        """Generate a default name for the experiment.
        Default name is the current date and time in the format of
        "YYYYMMDDHHMMSSffffff"

        Returns:
            str: generated default name
        """
        return datetime.now(TZ).strftime("%Y%m%d%H%M%S%f")

    @staticmethod
    def _check_json_extension(file_path: str | Path) -> None:
        if Path(file_path).suffix.lower() != ".json":
            raise InvalidFileExtensionError(f"File '{file_path}' does not have a '.json' extension.")

    def init(self) -> "Experiment":
        """Initialize the experiment directory and DB.

        Returns:
            Experiment: initialized experiment
        """
        if self.name is None:
            self.name = self._generate_default_name()
        if self.desc is None:
            self.desc = ""

        self.experiment_dirc = self.root_experiment_dirc / self.name
        # create experiment directory if it does not exist or empty
        if (not self.experiment_dirc.exists()) or (not any(self.experiment_dirc.iterdir())):
            self.experiment_dirc.mkdir(parents=True, exist_ok=True)
        else:
            raise ExperimentSettingError(f"Experiment directory '{self.experiment_dirc}' already exists.")

        self.exp_db = ExperimentDB(
            name=self.name,
            desc=self.desc,
            working_dirc=Path.cwd(),
            experiment_dirc=self.experiment_dirc,
            runs=[],
        )

        return self

    def load(self, exp_dirc: str | Path, exp_file_name: str | Path = DEFAULT_EXP_DB_FILE) -> "Experiment":
        """Load existing experiment data from a json file.

        Args:
            exp_dirc (str | Path): path to the experiment directory

        Raises:
            FileNotFoundError: if the experiment file does not exist
            ExperimentSettingError: if the experiment directory does not exist

        Returns:
            Experiment: loaded experiment
        """
        exp_file_path = Path(exp_dirc) / exp_file_name
        if not exp_file_path.exists():
            raise FileNotFoundError(f"{exp_file_path} does not exist.")

        self._check_json_extension(exp_file_path)
        with open(exp_file_path, "r") as json_file:
            exp_data = json.load(json_file)

        # set the experiment data from the json file
        self.exp_db = ExperimentDB(**exp_data)

        # update the experiment name, description, working directory, and experiment directory
        # if the loaded data is different from the current settings
        if (self.name is not None) and (self.name != self.exp_db.name):
            self.exp_db.name = self.name
            self.logger.info(f'Name is changed from "{self.exp_db.name}" to "{self.name}".')
        else:
            self.name = self.exp_db.name

        if (self.desc is not None) and (self.desc != self.exp_db.desc):
            self.exp_db.desc = self.desc
            self.logger.info(f'Description is changed from "{self.exp_db.desc}" to "{self.desc}".')
        else:
            self.desc = self.exp_db.desc

        working_dirc = Path.cwd()
        if working_dirc != self.exp_db.working_dirc:
            self.logger.info(f'Working directory is changed from "{self.exp_db.working_dirc}" to "{working_dirc}".')
            self.exp_db.working_dirc = working_dirc

        self.experiment_dirc = self.root_experiment_dirc / str(self.name)
        if self.experiment_dirc != self.exp_db.experiment_dirc:
            self.logger.info(
                f'Experiment directory is changed from "{self.exp_db.experiment_dirc}" to "{self.experiment_dirc}".'
            )
            self.exp_db.experiment_dirc = self.experiment_dirc

        if not self.exp_db.experiment_dirc.exists():
            raise ExperimentSettingError(f"Experiment directory '{self.exp_db.experiment_dirc}' does not exist.")

        self.current_run_id = len(self.exp_db.runs)
        self.save_experiment()

        return self

    def _is_initialized(self) -> None:
        """Check if the experiment is initialized.

        Raises:
            ExperimentNotInitializedError: if the experiment is not initialized
        """
        if self.exp_db is None:
            raise ExperimentNotInitializedError(
                "Experiment is not initialized. Please call init() or load_experiment() method first."
            )

    def _run_setup(self) -> Path:
        """Set up the directory structure for a new experiment run.

        This method creates a new run directory and updates the current run ID.
        The directory structure follows the pattern: experiment_dirc/run_{run_id}

        Returns:
            Path:
                Path to the newly created run directory.

        Raises:
            Exception:
                If an error occurs while creating the run directory.

        Note:
            - The run ID is automatically incremented.
            - If the run directory already exists, an error is raised.
            - The method ensures proper directory structure for storing run artifacts.
        """
        current_run_id = self.current_run_id + 1
        try:
            current_run_dirc = self.experiment_dirc / f"run_{current_run_id}"
            current_run_dirc.mkdir(parents=True)
        except Exception as e:
            self.logger.error(f"Error creating run directory: {e}")
            raise

        self.current_run_id = current_run_id

        return current_run_dirc

    def _run_backfill(self) -> None:
        """Clean up and revert changes when a run fails.

        This method handles the cleanup process when an error occurs during a run by:
        - Removing the partially created run directory
        - Reverting the current run ID to its previous value

        Note:
            - This method is called automatically when an exception occurs during run execution.
            - It ensures that failed runs don't leave behind incomplete artifacts.
            - The method maintains the integrity of the experiment's run sequence.
        """
        run_dirc = self.experiment_dirc / f"run_{self.current_run_id}"
        if run_dirc.exists():
            shutil.rmtree(run_dirc)

        self.current_run_id -= 1

    def run_evaluation(
        self,
        model_type: str,
        task_type: Optional[str],
        params: dict[str, Any],
        default_metrics_name: Optional[list[str]],
        custom_metrics: Optional[list[dict[str, Any]]],
    ) -> dict:
        """Run evaluation for the current run.

        Args:
            model_type (str): type of the model (qkernel or vqe)
            task_type (Optional[str]): type of the qkernel task (classification or regression)
            actual (np.ndarray): array of actual values
            predicted (np.ndarray): array of predicted values
            default_metrics_name (Optional[list[str]]): list of default metrics name
            custom_metrics (Optional[list[dict[str, Any]]]): list of user defined custom metric configurations

        Returns:
            dict: evaluation result
        """
        if model_type == QKERNEL_MODEL_TYPE_NAME and task_type == "classification":
            evaluation = ClassificationEvaluation(
                params=params,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        elif model_type == QKERNEL_MODEL_TYPE_NAME and task_type == "regression":
            evaluation = RegressionEvaluation(
                params=params,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        elif model_type == VQE_MODEL_TYPE_NAME:
            evaluation = VQEEvaluation(
                params=params,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}, task_type: {task_type}")

        evaluation.evaluate()

        return evaluation.to_dict()

    def _run_from_config(
        self,
        config: ExperimentConfig,
        commit_id: str,
        run_dirc: str | Path,
        n_jobs: int,
        show_progress: bool = True,
        repo_path: Optional[str] = None,
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Execute an experiment run using configuration settings.

        This method handles the execution of an experiment based on configuration settings, including:
        - Dataset and model initialization
        - Model training and evaluation
        - Result tracking and storage
        - Configuration file management

        Args:
            config (ExperimentConfig):
                Configuration settings for the experiment.
            commit_id (str):
                Git commit ID for version tracking.
            run_dirc (str | Path):
                Directory path for storing run results.
            n_jobs (int):
                Number of parallel jobs for processing.
            show_progress (bool, optional):
                Whether to display progress bars. Defaults to True.
            repo_path (Optional[str], optional):
                Path to git repository. Defaults to None.
            add_results (bool, optional):
                Whether to save run results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]:
                A tuple containing the run artifact and record.

        Raises:
            ValueError:
                If the model type specified in config is invalid.

        Note:
            - The method supports both QKernel and VQE model types.
            - The configuration file is saved in the run directory.
            - Model-specific evaluation metrics are automatically handled.
        """
        model_type = config.global_settings.model_type

        if model_type == QKERNEL_MODEL_TYPE_NAME:
            # create dataset instance from pre defined raw_preprocess_logic and transform_logic
            dataset = DatasetBuilder(config=config).build()

            # create model instance from the config
            model = ModelBuilder(config=config, n_jobs=n_jobs, show_progress=show_progress).build()
            save_shots_path = Path(run_dirc) / DEFAULT_SHOT_RESULTS_NAME if add_results else None
            save_model_path = Path(run_dirc) / DEFAULT_MODEL_NAME

            artifact, record = self._run_qkernel_from_instance(
                model_type=config.global_settings.model_type,
                task_type=config.global_settings.task_type,
                dataset=dataset,
                model=cast(BaseMLModel, model),
                save_shots_path=save_shots_path,
                save_model_path=save_model_path,
                default_metrics_name=config.evaluation.default_metrics,
                custom_metrics=config.evaluation.custom_metrics,
                desc=config.description,
                commit_id=commit_id,
                config_file_name=DEFAULT_EXP_CONFIG_FILE,
                repo_path=repo_path,
                add_results=add_results,
            )
        elif model_type == VQE_MODEL_TYPE_NAME:
            model = VQEBuilder(config=config, n_jobs=n_jobs).build()
            save_shots_path = Path(run_dirc) / DEFAULT_SHOT_RESULTS_NAME if add_results else None
            artifact, record = self._run_vqe_from_instance(
                model_type=config.global_settings.model_type,
                model=cast(BaseVQE, model),
                save_shots_path=save_shots_path,
                default_metrics_name=config.evaluation.default_metrics,
                custom_metrics=config.evaluation.custom_metrics,
                desc=config.description,
                commit_id=commit_id,
                config_file_name=DEFAULT_EXP_CONFIG_FILE,
                repo_path=repo_path,
                add_results=add_results,
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        return artifact, record

    def _get_auto_description(self, desc: str, repo_path: Optional[str] = None) -> str:
        """Generate an automatic description for a run if none is provided.

        This method handles automatic description generation when:
        - The description is empty
        - Auto-generation mode is enabled
        - Git is available for version tracking

        Args:
            desc (str):
                Current description of the run.
            repo_path (Optional[str], optional):
                Path to git repository. Defaults to None.

        Returns:
            str:
                Generated description if conditions are met, otherwise the original description.

        Note:
            - Requires git to be available for generating meaningful descriptions.
            - If git is not available, a warning is logged and the original description is returned.
            - The generated description includes information about code changes.
        """
        if self.auto_gen_mode and (desc == ""):
            if IS_GIT_AVAILABLE:
                desc = self.desc_generator.generate(
                    add_code=get_git_add_code(repo_path=repo_path, logger=self.logger),
                    remove_code=get_git_rm_code(repo_path=repo_path, logger=self.logger),
                )
            else:
                self.logger.warning(
                    """Git command is not available. DescriptionGenerator need git environment to generate.
                        Current run description set to empty string."""
                )
                desc = ""
        return desc

    def _run_qkernel_from_instance(
        self,
        model_type: str,
        task_type: Optional[str],
        dataset: Dataset,
        model: BaseMLModel,
        save_shots_path: Optional[str | Path],
        save_model_path: str | Path,
        default_metrics_name: Optional[list[str]],
        custom_metrics: Optional[list[dict[str, Any]]],
        desc: str,
        commit_id: str,
        config_file_name: Path,
        repo_path: Optional[str] = None,
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Execute a QKernel experiment run.

        This method handles the execution of a QKernel experiment, including:
        - Model training and prediction
        - Validation and test evaluation
        - Remote machine logging (if applicable)
        - Artifact and record creation

        Args:
            model_type (str):
                Type of the model (must be 'qkernel').
            task_type (Optional[str]):
                Type of the task ('classification' or 'regression').
            dataset (Dataset):
                Dataset instance containing training, validation, and test data.
            model (BaseMLModel):
                QKernel model instance to run.
            save_shots_path (Optional[str | Path]):
                Path to save shot results. If None, results are not saved.
            save_model_path (str | Path):
                Path to save the trained model.
            default_metrics_name (Optional[list[str]]):
                List of default metrics to evaluate.
            custom_metrics (Optional[list[dict[str, Any]]]):
                List of custom metric configurations.
            desc (str):
                Description of the run.
            commit_id (str):
                Git commit ID for version tracking.
            config_file_name (Path):
                Name of the configuration file.
            repo_path (Optional[str], optional):
                Path to git repository. Defaults to None.
            add_results (bool, optional):
                Whether to save run results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]:
                A tuple containing the run artifact and record.

        Raises:
            ValueError:
                If model_type is not 'qkernel' or task_type is invalid.

        Note:
            - The method supports both classification and regression tasks.
            - Validation metrics are only computed if validation data is provided.
            - Remote machine logging is automatically handled for quantum devices.
            - The method saves both the model and shot results if add_results is True.
        """
        train_start_dt = datetime.now(TZ)
        model.fit(X=dataset.X_train, y=dataset.y_train, save_shots_path=save_shots_path)
        train_end_dt = datetime.now(TZ)
        train_seconds = (train_end_dt - train_start_dt).total_seconds()

        if (dataset.X_val is not None) and (dataset.y_val is not None):
            validation_start_dt = datetime.now(TZ)
            validation_predicted = model.predict(dataset.X_val, bar_label="Validation")
            validation_end_dt = datetime.now(TZ)
            validation_seconds = (validation_end_dt - validation_start_dt).total_seconds()
            validation_evaluation = self.run_evaluation(
                model_type=model_type,
                task_type=task_type,
                params={"actual": dataset.y_val, "predicted": validation_predicted},
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        else:
            validation_seconds = None
            validation_evaluation = None

        test_start_dt = datetime.now(TZ)
        test_predicted = model.predict(dataset.X_test, bar_label="Test")
        test_end_dt = datetime.now(TZ)
        test_seconds = (test_end_dt - test_start_dt).total_seconds()
        test_evaluation = self.run_evaluation(
            model_type=model_type,
            task_type=task_type,
            params={"actual": dataset.y_test, "predicted": test_predicted},
            default_metrics_name=default_metrics_name,
            custom_metrics=custom_metrics,
        )

        # get remote quantum machine log
        device = cast(BaseKernelModel, model).kernel.device
        if device.is_remote():
            train_job_ids = device.get_job_ids(created_after=train_start_dt, created_before=train_end_dt)
            validation_job_ids = (
                device.get_job_ids(created_after=validation_start_dt, created_before=validation_end_dt)
                if validation_evaluation
                else []
            )
            test_job_ids = device.get_job_ids(created_after=test_start_dt, created_before=test_end_dt)
            remote_machine_log = RemoteMachine(
                provider=device.get_provider(),
                backend=device.get_backend_name(),
                job_ids=train_job_ids + validation_job_ids + test_job_ids,
            )
        else:
            remote_machine_log = None

        if add_results:
            model.save(save_model_path)

        artifact = RunArtifact(
            run_id=self.current_run_id,
            dataset=dataset,
            model=model,
        )

        record = RunRecord(
            run_id=self.current_run_id,
            desc=self._get_auto_description(desc, repo_path),
            remote_machine=remote_machine_log,
            commit_id=commit_id,
            config_file_name=config_file_name,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            runtime=RunTime(
                train_seconds=train_seconds,
                validation_seconds=validation_seconds,
                test_seconds=test_seconds,
            ),
            evaluations=Evaluations(validation=validation_evaluation, test=test_evaluation),
        )

        return artifact, record

    def _run_vqe_from_instance(
        self,
        model_type: str,
        model: BaseVQE,
        save_shots_path: Optional[str | Path],
        default_metrics_name: Optional[list[str]],
        custom_metrics: Optional[list[dict[str, Any]]],
        desc: str,
        commit_id: str,
        config_file_name: Path,
        repo_path: Optional[str] = None,
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Execute a VQE (Variational Quantum Eigensolver) experiment run.

        This method handles the execution of a VQE experiment, including:
        - Model optimization
        - Result evaluation
        - Artifact and record creation
        - Remote machine logging (if applicable)

        Args:
            model_type (str):
                Type of the model (must be 'vqe').
            model (BaseVQE):
                VQE model instance to run.
            save_shots_path (Optional[str | Path]):
                Path to save shot results. If None, results are not saved.
            default_metrics_name (Optional[list[str]]):
                List of default metrics to evaluate.
            custom_metrics (Optional[list[dict[str, Any]]]):
                List of custom metric configurations.
            desc (str):
                Description of the run.
            commit_id (str):
                Git commit ID for version tracking.
            config_file_name (Path):
                Name of the configuration file.
            repo_path (Optional[str], optional):
                Path to git repository. Defaults to None.
            add_results (bool, optional):
                Whether to save run results. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]:
                A tuple containing the run artifact and record.

        Raises:
            NotImplementedError:
                If remote machine execution is attempted (not yet supported for VQE).
            ValueError:
                If model_type is not 'vqe'.

        Note:
            - The method currently does not support remote machine execution.
            - Optimization parameters (init_params, max_steps) are hardcoded and may be
              moved to configuration in future versions.
            - The method automatically handles evaluation metrics for VQE models.
        """
        optimize_start_dt = datetime.now(TZ)
        # [TODO]: receive init_params from the config or argments
        model.optimize(init_params=None, max_steps=20, verbose=True)
        optimize_end_dt = datetime.now(TZ)
        optimize_seconds = (optimize_end_dt - optimize_start_dt).total_seconds()
        evaluations = self.run_evaluation(
            model_type=model_type,
            task_type=None,
            params={"cost_history": model.cost_history, "hamiltonian": model.hamiltonian},
            default_metrics_name=default_metrics_name,
            custom_metrics=custom_metrics,
        )

        # get remote quantum machine log
        if model.device.is_remote():
            raise NotImplementedError("Remote machine is not supported for VQE.")
        else:
            remote_machine_log = None

        artifact = RunArtifact(
            run_id=self.current_run_id,
            dataset=None,
            model=model,
        )

        record = RunRecord(
            run_id=self.current_run_id,
            desc=self._get_auto_description(desc, repo_path),
            remote_machine=remote_machine_log,
            commit_id=commit_id,
            config_file_name=config_file_name,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            runtime=VQERunTime(
                optimize_seconds=optimize_seconds,
            ),
            evaluations=VQEEvaluations(optimized=evaluations),
        )

        return artifact, record

    def run(
        self,
        model_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        model: Optional[BaseMLModel | BaseVQE] = None,
        config_source: Optional[ExperimentConfig | str | Path] = None,
        default_metrics_name: Optional[list[str]] = None,
        custom_metrics: Optional[list[dict[str, Any]]] = None,
        n_jobs: int = DEFAULT_N_JOBS,
        show_progress: bool = True,
        desc: str = "",
        repo_path: Optional[str] = None,
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Execute a new experiment run with the specified configuration.

        This method supports two main modes of operation:
        1. Config-based mode: Using a configuration file or ExperimentConfig instance
        2. Instance-based mode: Directly providing dataset and model instances

        The method handles:
        - Experiment setup and directory creation
        - Model training and evaluation
        - Result tracking and storage
        - Artifact management
        - Git integration (if available)

        Args:
            model_type (Optional[str], optional):
                Type of model to use ('qkernel' or 'vqe').
                Required for instance-based mode. Defaults to None.
            task_type (Optional[str], optional):
                Type of task for QKernel models ('classification' or 'regression').
                Required for QKernel models. Defaults to None.
            dataset (Optional[Dataset], optional):
                Dataset instance for instance-based mode. Defaults to None.
            model (Optional[BaseMLModel | BaseVQE], optional):
                Model instance for instance-based mode. Defaults to None.
            config_source (Optional[ExperimentConfig | str | Path], optional):
                Configuration source for config-based mode.
                Can be an ExperimentConfig instance or path to config file.
                Defaults to None.
            default_metrics_name (Optional[list[str]], optional):
                List of default metrics to evaluate. Defaults to None.
            custom_metrics (Optional[list[dict[str, Any]]], optional):
                List of custom metric configurations. Defaults to None.
            n_jobs (int, optional):
                Number of parallel jobs for processing. Defaults to DEFAULT_N_JOBS.
            show_progress (bool, optional):
                Whether to display progress bars. Defaults to True.
            desc (str, optional):
                Description of the run. Defaults to "".
            repo_path (Optional[str], optional):
                Path to git repository for version tracking. Defaults to None.
            add_results (bool, optional):
                Whether to save run results and artifacts. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]:
                A tuple containing the run artifact and record.

        Raises:
            ExperimentNotInitializedError:
                If the experiment has not been initialized.
            ExperimentRunSettingError:
                If required parameters are missing or invalid.
            ValueError:
                If model_type is invalid.

        Note:
            - For config-based mode, the configuration file will be saved in the run directory.
            - For instance-based mode, model_type must be specified.
            - If git is available, the commit ID will be tracked.
        """
        self._is_initialized()

        if add_results:
            current_run_dirc = self._run_setup()
            commit_id = get_commit_id(repo_path=repo_path) if IS_GIT_AVAILABLE else ""
        else:
            current_run_dirc = Path("")
            commit_id = ""

        try:
            if config_source is not None:
                if isinstance(config_source, str | Path):
                    config = ExperimentConfig(path=config_source)
                else:
                    config = config_source

                artifact, record = self._run_from_config(
                    config=config,
                    commit_id=commit_id,
                    run_dirc=current_run_dirc,
                    n_jobs=n_jobs,
                    show_progress=show_progress,
                    repo_path=repo_path,
                    add_results=add_results,
                )
            elif (dataset is not None) and (model is not None):
                if model_type is None:
                    raise ExperimentRunSettingError(
                        f"""
                        model_type must be provided when dataset and model are provided.
                        Please provide model_type={QKERNEL_MODEL_TYPE_NAME} or {VQE_MODEL_TYPE_NAME}.
                        """
                    )
                elif model_type == QKERNEL_MODEL_TYPE_NAME:
                    artifact, record = self._run_qkernel_from_instance(
                        model_type=model_type,
                        task_type=task_type,
                        dataset=dataset,
                        model=cast(BaseMLModel, model),
                        save_shots_path=current_run_dirc / DEFAULT_SHOT_RESULTS_NAME if add_results else None,
                        save_model_path=current_run_dirc / DEFAULT_MODEL_NAME,
                        default_metrics_name=default_metrics_name,
                        custom_metrics=custom_metrics,
                        desc=desc,
                        commit_id=commit_id,
                        config_file_name=Path(""),
                        repo_path=repo_path,
                        add_results=add_results,
                    )
                elif model_type == VQE_MODEL_TYPE_NAME:
                    artifact, record = self._run_vqe_from_instance(
                        model_type=model_type,
                        model=cast(BaseVQE, model),
                        save_shots_path=current_run_dirc / DEFAULT_SHOT_RESULTS_NAME if add_results else None,
                        default_metrics_name=default_metrics_name,
                        custom_metrics=custom_metrics,
                        desc=desc,
                        commit_id=commit_id,
                        config_file_name=Path(""),
                        repo_path=repo_path,
                        add_results=add_results,
                    )
                else:
                    raise ValueError(f"Invalid model_type: {model_type}")
            else:
                raise ExperimentRunSettingError("Either config or dataset and model must be provided.")
        except Exception as e:
            self.logger.error(f"Error occurred during the run: {e}")
            if add_results:
                self._run_backfill()
            raise e

        if add_results:
            self.exp_db.runs.append(record)  # type: ignore
            self.save_experiment()

            if config_source is not None:
                # config is saved in the run directory
                # if "run" executed by instance mode, config is not saved
                save_experiment_config_to_yaml(
                    config, Path(current_run_dirc) / DEFAULT_EXP_CONFIG_FILE, delete_source_path=True
                )

        return artifact, record

    def runs_to_dataframe(self, include_validation: bool = False) -> pd.DataFrame:
        """Convert experiment run records into a pandas DataFrame.

        This method transforms the experiment's run records into a structured DataFrame,
        making it easier to analyze and compare results across different runs.
        The DataFrame includes:
        - Run IDs
        - Evaluation metrics for test data
        - Optional validation metrics (if include_validation is True)

        Args:
            include_validation (bool, optional):
                Whether to include validation metrics in the DataFrame.
                Defaults to False.

        Returns:
            pd.DataFrame:
                A DataFrame containing run results with the following columns:
                - run_id: The ID of each run
                - [metric_name]: Evaluation metrics for test data
                - [metric_name]_validation: Validation metrics (if include_validation is True)

        Raises:
            ExperimentNotInitializedError:
                If the experiment has not been initialized.
            ValueError:
                If the run records contain invalid evaluation types.

        Note:
            - The DataFrame will be empty if no runs have been recorded.
            - Validation metrics are only included if they were computed during the run.
            - The method automatically handles both QKernel and VQE evaluation types.
        """
        self._is_initialized()
        if self.exp_db.runs is None:  # type: ignore
            return pd.DataFrame()

        if isinstance(cast(ExperimentDB, self.exp_db).runs[0].evaluations, Evaluations):
            evaluation_key = "test"
        elif isinstance(cast(ExperimentDB, self.exp_db).runs[0].evaluations, VQEEvaluations):
            evaluation_key = "optimized"
        else:
            raise ValueError(
                f"Invalid run_record.evaluations: {type(cast(ExperimentDB, self.exp_db).runs[0].evaluations)}"
            )

        run_data = [run.model_dump() for run in self.exp_db.runs]  # type: ignore
        run_data = [
            {
                "run_id": run_record_dict["run_id"],
                **run_record_dict["evaluations"][evaluation_key],
                **(
                    {
                        f"{key}_validation": value
                        for key, value in (run_record_dict["evaluations"]["validation"] or {}).items()
                    }
                    if include_validation and "validation" in run_record_dict.get("evaluations", {})
                    else {}
                ),
            }
            for run_record_dict in run_data
        ]
        return pd.DataFrame(run_data)

    def save_experiment(self, exp_file: str | Path = DEFAULT_EXP_DB_FILE) -> None:
        """Save the experiment data to a json file.

        Args:
            exp_file (str | Path, optional):
                name of the file to save the experiment data.Defaults to DEFAULT_EXP_DB_FILE.

        Raises:
            ExperimentNotInitializedError: if the experiment is not initialized
        """

        def custom_encoder(obj: Any) -> str:
            if isinstance(obj, Path):
                return str(obj)
            raise JsonEncodingError(f"Object of type {type(obj).__name__} is not JSON serializable")

        self._is_initialized()
        save_path = self.experiment_dirc / exp_file
        self._check_json_extension(save_path)
        exp_data = json.loads(self.exp_db.model_dump_json())  # type: ignore
        with open(save_path, "w") as json_file:
            json.dump(exp_data, json_file, indent=4, default=custom_encoder)

    def get_run_record(self, runs: list[RunRecord], run_id: int) -> RunRecord:
        """Get the run record of the target run_id.

        Args:
            run_id (int): target run_id

        Raises:
            ValueError: if the run record does not exist

        Returns:
            RunRecord: target run record
        """
        self._is_initialized()
        for run_record in runs:
            if run_record.run_id == run_id:
                return run_record

        # if the target run_id does not exist
        raise ValueError(f"Run record of run_id={run_id} does not exist.")

    def _validate_evaluation(
        self, logging_evaluation: dict[str, float], reproduction_evaluation: dict[str, float], tol: float
    ) -> None:
        """Validate the consistency between original and reproduced evaluation results.

        This method compares evaluation metrics between the original run and its reproduction,
        ensuring that the results are within the specified tolerance.

        Args:
            logging_evaluation (dict[str, float]):
                Evaluation metrics from the original run.
            reproduction_evaluation (dict[str, float]):
                Evaluation metrics from the reproduced run.
            tol (float):
                Tolerance threshold for comparing metrics.

        Raises:
            ValueError:
                If any metric differs beyond the specified tolerance or if metrics are missing.

        Note:
            - The comparison is performed for all metrics in the original evaluation.
            - Missing metrics in the reproduction are treated as errors.
            - The tolerance is applied to the absolute difference between values.
        """
        invalid_dict: dict[str, str] = {}
        for key, value in logging_evaluation.items():
            reproduction_value = reproduction_evaluation.get(key, None)
            if reproduction_value is None:
                raise ValueError(f"Reproduction evaluation of {key} is not found.")
            elif abs(value - reproduction_value) > tol:
                invalid_dict[key] = f"{value} -> {reproduction_value}"

        if len(invalid_dict) > 0:
            raise ValueError(
                f"Evaluation results are different between logging and reproduction (invalid metrics: {invalid_dict})."
            )

    def reproduce(self, run_id: int, check_commit_id: bool = False, tol: float = 1e-6) -> tuple[RunArtifact, RunRecord]:
        """Reproduce a previous experiment run using its configuration.

        This method recreates a previous experiment run by:
        1. Loading the original configuration
        2. Re-running the experiment with the same settings
        3. Validating the results against the original run
        4. Optionally checking git commit consistency

        Args:
            run_id (int):
                ID of the run to reproduce.
            check_commit_id (bool, optional):
                Whether to verify that the current git commit matches the original run.
                Defaults to False.
            tol (float, optional):
                Tolerance for comparing evaluation metrics between original and reproduced runs.
                Defaults to 1e-6.

        Returns:
            tuple[RunArtifact, RunRecord]:
                A tuple containing the reproduced run's artifact and record.

        Raises:
            ExperimentNotInitializedError:
                If the experiment has not been initialized.
            ReproductionError:
                If the run was executed in instance-based mode (no config file available).
            ValueError:
                If the reproduced results differ significantly from the original run.

        Note:
            - Only runs executed in config-based mode can be reproduced.
            - The method will raise an error if the reproduced results differ from the original
              beyond the specified tolerance.
            - If check_commit_id is True and the current commit differs from the original,
              a warning will be logged but the reproduction will continue.
        """
        self._is_initialized()
        run_record = self.get_run_record(self.exp_db.runs, run_id)  # type: ignore

        if check_commit_id:
            commit_id = get_commit_id() if IS_GIT_AVAILABLE else ""
            if commit_id != run_record.commit_id:
                self.logger.warning(
                    f'Current commit_id="{commit_id}" is different from'
                    f'the run_id={run_id} commit_id="{run_record.commit_id}".'
                )
        if run_record.config_file_name == Path(""):
            raise ReproductionError(
                f"run_id={run_id} does not have a config file path. This run executed from instance."
                "Run from instance mode not supported for reproduction."
            )

        config_path = Path(f"{self.experiment_dirc}/run_{run_id}/{DEFAULT_EXP_CONFIG_FILE}")
        reproduced_artifact, reproduced_result = self.run(config_source=config_path, add_results=False)

        if isinstance(run_record.evaluations, Evaluations) and isinstance(reproduced_result.evaluations, Evaluations):
            logging_evaluation = run_record.evaluations.test
            reproduced_evaluation = reproduced_result.evaluations.test
        elif isinstance(run_record.evaluations, VQEEvaluations) and isinstance(
            reproduced_result.evaluations, VQEEvaluations
        ):
            logging_evaluation = run_record.evaluations.optimized
            reproduced_evaluation = reproduced_result.evaluations.optimized
        else:
            raise ValueError(
                f"""Invalid run_record.evaluations: {type(run_record.evaluations)},
                reproduced_result.evaluations: {type(reproduced_result.evaluations)}"""
            )

        self._validate_evaluation(logging_evaluation, reproduced_evaluation, tol=tol)
        self.logger.info(f"Reproduce model is successful. Evaluation results are the same as run_id={run_id}.")

        return reproduced_artifact, reproduced_result

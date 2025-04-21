import os
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
from qxmt.datasets import Dataset
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    ExperimentSettingError,
)
from qxmt.experiment.executor import QKernelExecutor, VQEExecutor
from qxmt.experiment.repository import ExperimentRepository
from qxmt.experiment.reproducer import Reproducer
from qxmt.experiment.schema import (
    Evaluations,
    ExperimentDB,
    RunArtifact,
    RunRecord,
    VQEEvaluations,
)
from qxmt.logger import set_default_logger
from qxmt.models.qkernels import BaseMLModel
from qxmt.models.vqe import BaseVQE
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
    """Manage the life-cycle of quantum machine-learning experiments.

    The Experiment class is the centerpiece of the *Quantum eXperiment Management Tool* (QXMT).
    It orchestrates creation and loading of experiment directories, execution of runs,
    persistence and aggregation of results, and ensures reproducibility. Every
    side-effect that touches the file-system is delegated to
    :class:`qxmt.experiment.repository.ExperimentRepository`, so this class can focus on
    business logic.

    Responsibilities:

    1. **Initialization / Loading**

       * :py:meth:`init` - create a new experiment directory and an empty `ExperimentDB`.
       * :py:meth:`load` - load an existing experiment from a JSON file.

    2. **Run management**

       * :py:meth:`run` - execute an experiment run with QKernel or VQE models (supports both
         config-based and instance-based workflows).
       * :py:meth:`_run_setup` / :py:meth:`_run_backfill` - create and rollback run directories.

    3. **Result handling**

       * :py:meth:`runs_to_dataframe` - convert `RunRecord`s into a `pandas.DataFrame` for easy analysis.
       * :py:meth:`save_experiment` - persist the `ExperimentDB` to disk.
       * :py:meth:`get_run_record` - retrieve a single `RunRecord` by *run_id*.

    4. **Reproducibility**

       * :py:meth:`reproduce` - re-execute a past run and validate that results match.

    Example:

        .. code-block:: python

            from qxmt.experiment import Experiment

            exp = Experiment(name="my_exp").init()
            artifact, record = exp.run(
                model_type="qkernel",
                task_type="classification",
                dataset=dataset_instance,
                model=model_instance,
            )

            # Aggregate results
            df = exp.runs_to_dataframe()

    Attributes:
        name (str | None):
            Experiment name. If *None*, a timestamped name is generated.
        desc (str | None):
            Human-readable description. When `auto_gen_mode` is enabled and the value is an
            empty string, an LLM can generate it automatically.
        auto_gen_mode (bool):
            Whether to generate run descriptions with an LLM.
        root_experiment_dirc (pathlib.Path):
            Root directory where all experiments are stored.
        experiment_dirc (pathlib.Path):
            Directory assigned to this particular experiment.
        current_run_id (int):
            ID of the next run to be executed (zero-based).
        exp_db (ExperimentDB | None):
            In-memory database object holding experiment meta-data.
        logger (logging.Logger):
            Logger instance to report progress and warnings.
        _repo (ExperimentRepository):
            Internal repository that encapsulates all filesystem & persistence operations.
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
            name (Optional[str]): Name of the experiment. If None, a default name will be generated
                using the current timestamp. Defaults to None.
            desc (Optional[str]): Description of the experiment. Used for documentation and search purposes.
                If None and auto_gen_mode is True, a description will be generated. Defaults to None.
            auto_gen_mode (bool): Whether to use the DescriptionGenerator for automatic description generation.
                Requires the "USE_LLM" environment variable to be set to True. Defaults to USE_LLM.
            root_experiment_dirc (str | Path): Root directory where experiment data will be stored.
                Defaults to DEFAULT_EXP_DIRC.
            llm_model_path (str): Path to the LLM model used for description generation.
                Defaults to LLM_MODEL_PATH.
            logger (Logger): Logger instance for handling warning and error messages.
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
        # Repository handles all filesystem & persistence side‑effects
        self._repo = ExperimentRepository(logger)

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

        Default name is the current date and time in the format of "YYYYMMDDHHMMSSffffff".

        Returns:
            str: Generated default name.
        """
        return datetime.now(TZ).strftime("%Y%m%d%H%M%S%f")

    def init(self) -> "Experiment":
        """Initialize the experiment directory and DB.

        Creates a new experiment directory and initializes an empty ExperimentDB.
        The directory will be created under root_experiment_dirc with the experiment name.

        Returns:
            Experiment: Initialized experiment instance.

        Raises:
            ExperimentSettingError: If the experiment directory already exists.
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
            exp_dirc (str | Path): Path to the experiment directory.
            exp_file_name (str | Path): Name of the experiment file. Defaults to DEFAULT_EXP_DB_FILE.

        Returns:
            Experiment: Loaded experiment instance.

        Raises:
            FileNotFoundError: If the experiment file does not exist.
            ExperimentSettingError: If the experiment directory does not exist.
        """
        exp_file_path = Path(exp_dirc) / exp_file_name
        if not exp_file_path.exists():
            raise FileNotFoundError(f"{exp_file_path} does not exist.")

        # Load via repository
        self.exp_db = self._repo.load(exp_file_path)

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
            ExperimentNotInitializedError: If the experiment is not initialized.
        """
        if self.exp_db is None:
            raise ExperimentNotInitializedError(
                "Experiment is not initialized. Please call init() or load_experiment() method first."
            )

    def _run_setup(self) -> Path:
        """Set up the directory structure for a new experiment run.

        Creates a new run directory and updates the current run ID.
        The directory structure follows the pattern: experiment_dirc/run_{run_id}

        Returns:
            Path: Path to the newly created run directory.

        Raises:
            Exception: If an error occurs while creating the run directory.
        """
        # Delegate to repository – keeps Experiment free from FS details
        current_run_id = self.current_run_id + 1
        current_run_dirc = self._repo.create_run_dir(self.experiment_dirc, current_run_id)
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
        # Repository handles rollback deletion
        self._repo.remove_run_dir(self.experiment_dirc, self.current_run_id)
        self.current_run_id -= 1

    def _get_auto_description(self, desc: str, repo_path: Optional[str] = None) -> str:
        """Generate an automatic description for a run if none is provided.

        Args:
            desc (str): Current description of the run.
            repo_path (Optional[str]): Path to git repository. Defaults to None.

        Returns:
            str: Generated description if conditions are met, otherwise the original description.

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

        Args:
            model_type (Optional[str]): Type of model to use ('qkernel' or 'vqe').
                Required for instance-based mode. Defaults to None.
            task_type (Optional[str]): Type of task for QKernel models ('classification' or 'regression').
                Required for QKernel models. Defaults to None.
            dataset (Optional[Dataset]): Dataset instance for instance-based mode. Defaults to None.
            model (Optional[BaseMLModel | BaseVQE]): Model instance for instance-based mode. Defaults to None.
            config_source (Optional[ExperimentConfig | str | Path]): Configuration source for config-based mode.
                Can be an ExperimentConfig instance or path to config file. Defaults to None.
            default_metrics_name (Optional[list[str]]): List of default metrics to evaluate. Defaults to None.
            custom_metrics (Optional[list[dict[str, Any]]]): List of custom metric configurations. Defaults to None.
            n_jobs (int): Number of parallel jobs for processing. Defaults to DEFAULT_N_JOBS.
            show_progress (bool): Whether to display progress bars. Defaults to True.
            desc (str): Description of the run. Defaults to "".
            repo_path (Optional[str]): Path to git repository for version tracking. Defaults to None.
            add_results (bool): Whether to save run results and artifacts. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: A tuple containing the run artifact and record.

        Raises:
            ExperimentNotInitializedError: If the experiment has not been initialized.
            ExperimentRunSettingError: If required parameters are missing or invalid.
            ValueError: If model_type is invalid.

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
            # ------------------------------------------------------------------
            # Config‑based mode
            # ------------------------------------------------------------------
            if config_source is not None:
                if isinstance(config_source, (str, Path)):
                    config = ExperimentConfig(path=config_source)
                else:
                    config = config_source

                model_type_cfg = config.global_settings.model_type
                if model_type_cfg == QKERNEL_MODEL_TYPE_NAME:
                    executor = QKernelExecutor(self)
                    artifact, record = executor.run_from_config(
                        config=config,
                        commit_id=commit_id,
                        run_dirc=current_run_dirc,
                        n_jobs=n_jobs,
                        show_progress=show_progress,
                        repo_path=repo_path,
                        add_results=add_results,
                    )
                elif model_type_cfg == VQE_MODEL_TYPE_NAME:
                    executor = VQEExecutor(self)
                    artifact, record = executor.run_from_config(
                        config=config,
                        commit_id=commit_id,
                        run_dirc=current_run_dirc,
                        n_jobs=n_jobs,
                        show_progress=show_progress,
                        repo_path=repo_path,
                        add_results=add_results,
                    )
                else:
                    raise ValueError(f"Invalid model_type: {model_type_cfg}")

            # ------------------------------------------------------------------
            # Instance‑based mode
            # ------------------------------------------------------------------
            elif (model_type == QKERNEL_MODEL_TYPE_NAME) and (dataset is not None) and (model is not None):
                executor = QKernelExecutor(self)
                artifact, record = executor.run_from_instance(
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
            elif (model_type == VQE_MODEL_TYPE_NAME) and (model is not None):
                executor = VQEExecutor(self)
                artifact, record = executor.run_from_instance(
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

        Args:
            include_validation (bool): Whether to include validation metrics in the DataFrame.
                Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing run results with the following columns:
                - run_id: The ID of each run
                - [metric_name]: Evaluation metrics for test data
                - [metric_name]_validation: Validation metrics (if include_validation is True)

        Raises:
            ExperimentNotInitializedError: If the experiment has not been initialized.
            ValueError: If the run records contain invalid evaluation types.

        Note:
            - The DataFrame will be empty if no runs have been recorded.
            - Validation metrics are only included if they were computed during the run.
            - The method automatically handles both QKernel and VQE evaluation types.
        """
        self._is_initialized()
        if not self.exp_db.runs:  # type: ignore
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
            exp_file (str | Path): Name of the file to save the experiment data.
                Defaults to DEFAULT_EXP_DB_FILE.

        Raises:
            ExperimentNotInitializedError: If the experiment is not initialized.
        """
        save_path = self.experiment_dirc / exp_file
        self._repo.save(self.exp_db, save_path)  # type: ignore[arg-type]

    def get_run_record(self, runs: list[RunRecord], run_id: int) -> RunRecord:
        """Get the run record of the target run_id.

        Args:
            runs (list[RunRecord]): List of run records to search.
            run_id (int): Target run_id.

        Returns:
            RunRecord: Target run record.

        Raises:
            ValueError: If the run record does not exist.
        """
        self._is_initialized()
        for run_record in runs:
            if run_record.run_id == run_id:
                return run_record

        # if the target run_id does not exist
        raise ValueError(f"Run record of run_id={run_id} does not exist.")

    def reproduce(self, run_id: int, check_commit_id: bool = False, tol: float = 1e-6) -> tuple[RunArtifact, RunRecord]:
        """Reproduce a previous experiment run using its configuration.

        Args:
            run_id (int): ID of the run to reproduce.
            check_commit_id (bool): Whether to verify that the current git commit matches the original run.
                Defaults to False.
            tol (float): Tolerance for comparing evaluation metrics between original and reproduced runs.
                Defaults to 1e-6.

        Returns:
            tuple[RunArtifact, RunRecord]: A tuple containing the reproduced run's artifact and record.

        Raises:
            ExperimentNotInitializedError: If the experiment has not been initialized.
            ReproductionError: If the run was executed in instance-based mode (no config file available).
            ValueError: If the reproduced results differ significantly from the original run.

        Note:
            - Only runs executed in config-based mode can be reproduced.
            - The method will raise an error if the reproduced results differ from the original
              beyond the specified tolerance.
            - If check_commit_id is True and the current commit differs from the original,
              a warning will be logged but the reproduction will continue.
        """
        return Reproducer(self).reproduce(run_id, check_commit_id=check_commit_id, tol=tol)

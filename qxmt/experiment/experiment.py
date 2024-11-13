import json
import os
import shutil
import time
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from qxmt.configs import ExperimentConfig
from qxmt.constants import (
    DEFAULT_EXP_CONFIG_FILE,
    DEFAULT_EXP_DB_FILE,
    DEFAULT_EXP_DIRC,
    DEFAULT_MODEL_NAME,
    DEFAULT_SHOT_RESULTS_NAME,
    LLM_MODEL_PATH,
    TZ,
)
from qxmt.datasets.builder import DatasetBuilder
from qxmt.datasets.schema import Dataset
from qxmt.evaluation.evaluation import ClassificationEvaluation, RegressionEvaluation
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    ExperimentSettingError,
    InvalidFileExtensionError,
    JsonEncodingError,
    ReproductionError,
)
from qxmt.experiment.schema import ExperimentDB, RunArtifact, RunRecord, RunTime
from qxmt.logger import set_default_logger
from qxmt.models.base import BaseMLModel
from qxmt.models.builder import ModelBuilder
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


class Experiment:
    """Experiment class for managing the experiment and each run data.
    The Experiment class provides methods for initializing the experiment,
    running the experiment, saving the experiment data, and reproducing the model.

    All experiment data is stored in the ExperimentDB instance.
    It is save in local directory as a json file (root_experiment_dirc/experiments/your_exp_name/experiment.json).

    Experiment can be initialized and strated from scratch by calling the `init()` method.
    Anthoer way is to load the existing experiment data from the json file (experiment.json)
    by calling the `load_experiment()` method.

    The Experiment class can be used in two ways:
    1. **Provide config_path**:
    This method accepts the path to the config file or config instance.
    It is more flexible but requires a YAML base config file.
    This method tracks the experiment settings, result and can reproduce the model.
    Officially, we recommend using the config file method.

    2. **Directly provide dataset and model instance**:
    This method directly accepts dataset and model instances.
    It is easy to use but does "NOT" track the experiment settings.
    This method is useful for adhoc experiments, quick testing or debugging.

    Examples:
        >>> import qxmt
        >>> exp = qxmt.Experiment(
        ...        name="my_qsvm_algorithm",
        ...       desc=\"""This is a experiment for new qsvm algorithm.
        ...        This experiment is applied and evaluated on multiple datasets.
        ...        \""",
        ...        auto_gen_mode=True,
        ...    ).init()
        >>> config_path = "../configs/template.yaml"
        >>> artifact, result = exp.run(
        ...     config_source=config_path)
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
        """Initialize the Experiment class.
        Set the experiment name, description,
        and other settings such as auto_gen_mode, root_experiment_dirc and logger.
        auto_gen_mode controls whether to use the DescriptionGenerator by LLM.
        If use, set environemnt variable "USE_LLM" to True.
        root_experiment_dirc is the root directory to save the experiment data.
        Each artifact and result store in the subdirectory of the root directory.

        Args:
            name (Optional[str], optional):
                experiment name. If None, generate by execution time. Defaults to None.
            desc (Optional[str], optional):
                description of the experiment.
                The purpose is search, memo, etc not used in the code. Defaults to None.
            auto_gen_mode (bool, optional):
                whether to use the DescriptionGenerator
                for generating the description of each run. Defaults to USE_LLM.
            root_experiment_dirc (str | Path, optional):
                root directory to save the experiment data. Defaults to DEFAULT_EXP_DIRC.
            llm_model_path (str, optional):
                path to the LLM model. Defaults to LLM_MODEL_PATH.
            logger (Logger, optional):
                logger instance for warning or error messages. Defaults to LOGGER.


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
        """Setup for the current run.
        Create a new run directory and update the current run ID.
        If the run directory already exists, raise an error and not update the current run ID.

        Returns:
            Path: path to the current run directory

        Raises:
            Exception: if error occurs while creating the run directory
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
        """Backfill the current run when an error occurs during the run."""
        run_dirc = self.experiment_dirc / f"run_{self.current_run_id}"
        if run_dirc.exists():
            shutil.rmtree(run_dirc)

        self.current_run_id -= 1

    def run_evaluation(
        self,
        task_type: str,
        actual: np.ndarray,
        predicted: np.ndarray,
        default_metrics_name: Optional[list[str]],
        custom_metrics: Optional[list[dict[str, Any]]],
    ) -> dict:
        """Run evaluation for the current run.

        Args:
            actual (np.ndarray): array of actual values
            predicted (np.ndarray): array of predicted values
            default_metrics_name (Optional[list[str]]): list of default metrics name
            custom_metrics (Optional[list[dict[str, Any]]]): list of user defined custom metric configurations

        Returns:
            dict: evaluation result
        """
        match task_type:
            case "classification":
                evaluation = ClassificationEvaluation(
                    actual=actual,
                    predicted=predicted,
                    default_metrics_name=default_metrics_name,
                    custom_metrics=custom_metrics,
                )
            case "regression":
                evaluation = RegressionEvaluation(
                    actual=actual,
                    predicted=predicted,
                    default_metrics_name=default_metrics_name,
                    custom_metrics=custom_metrics,
                )
            case _:
                raise ValueError(f"Invalid task_type: {task_type}")

        evaluation.evaluate()

        return evaluation.to_dict()

    def _run_from_config(
        self,
        config: ExperimentConfig,
        commit_id: str,
        run_dirc: str | Path,
        repo_path: Optional[str] = None,
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """Run the experiment from the config file.

        Args:
            config (ExperimentConfig): configuration of the experiment
            commit_id (str): commit ID of the current git repository
            run_dirc (str | Path): path to the run directory
            repo_path (str, optional): path to the git repository. Defaults to None.
            add_results (bool, optional): whether to save the model. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: artifact and record of the current run_id
        """
        # create dataset instance from pre defined raw_preprocess_logic and transform_logic
        dataset = DatasetBuilder(config=config).build()

        # create model instance from the config
        model = ModelBuilder(config=config).build()
        save_shots_path = Path(run_dirc) / DEFAULT_SHOT_RESULTS_NAME if add_results else None
        save_model_path = Path(run_dirc) / DEFAULT_MODEL_NAME

        artifact, record = self._run_from_instance(
            task_type=config.global_settings.task_type,
            dataset=dataset,
            model=model,
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

        return artifact, record

    def _run_from_instance(
        self,
        task_type: str,
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
        """Run the experiment from the dataset and model instance.

        Args:
            task_type (str): type of the task (classification or regression)
            dataset (Dataset): dataset object
            model (BaseMLModel): model object
            save_model_path (str | Path): path to save the model
            default_metrics_name (Optional[list[str]]): list of default metrics name
            custom_metrics (Optional[list[dict[str, Any]]]): list of user defined custom metric configurations
            desc (str, optional): description of the run.
            commit_id (str): commit ID of the current git repository
            config_file_name (Path): name of the config file
            repo_path (str, optional): path to the git repository. Defaults to None.
            add_results (bool, optional): whether to save the model. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: artifact and record of the current run_id
        """
        fit_start = time.perf_counter()
        model.fit(X=dataset.X_train, y=dataset.y_train, save_shots_path=save_shots_path)
        fit_end = time.perf_counter()

        predict_start = time.perf_counter()
        predicted = model.predict(dataset.X_test)
        predict_end = time.perf_counter()

        if add_results:
            model.save(save_model_path)

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

        artifact = RunArtifact(
            run_id=self.current_run_id,
            dataset=dataset,
            model=model,
        )

        record = RunRecord(
            run_id=self.current_run_id,
            desc=desc,
            commit_id=commit_id,
            config_file_name=config_file_name,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            runtime=RunTime(fit_seconds=fit_end - fit_start, predict_seconds=predict_end - predict_start),
            evaluation=self.run_evaluation(
                task_type=task_type,
                actual=dataset.y_test,
                predicted=predicted,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            ),
        )

        return artifact, record

    def run(
        self,
        task_type: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        model: Optional[BaseMLModel] = None,
        config_source: Optional[ExperimentConfig | str | Path] = None,
        default_metrics_name: Optional[list[str]] = None,
        custom_metrics: Optional[list[dict[str, Any]]] = None,
        desc: str = "",
        repo_path: Optional[str] = None,
        add_results: bool = True,
    ) -> tuple[RunArtifact, RunRecord]:
        """
        Start a new run for the experiment.

        The `run()` method can be called in two ways:

        1. **Provide dataset and model instance**:
        This method directly accepts dataset and model instances.
        It is easy to use but less flexible and does "NOT" track the experiment settings.

        2. **Provide config_path**:
        This method accepts the path to the config file or config instance.
        It is more flexible but requires a config file.

        Args:
            task_type (str, optional): type of the task (classification or regression). Defaults to None.
            dataset (Dataset): the dataset object.
            model (BaseMLModel): the model object.
            config_source (ExperimentConfig, str | Path, optional): config source can be either an `ExperimentConfig`
                instance or the path to a config file. If a path is provided, it loads and creates an
                `ExperimentConfig` instance. Defaults to None.
            default_metrics_name (list[str], optional): list of default metrics names. Defaults to None.
            custom_metrics (list[dict[str, Any]], optional):
                list of user defined custom metric configurations. Defaults to None.
            desc (str, optional): description of the run. Defaults to "".
            repo_path (str, optional): path to the git repository. Defaults to None.
            add_results (bool, optional): whether to add the run record to the experiment. Defaults to True.

        Returns:
            tuple[RunArtifact, RunRecord]: Returns a tuple containing the artifact and run record of the current run_id.

        Raises:
            ExperimentNotInitializedError: Raised if the experiment is not initialized.
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
                    add_results=add_results,
                )
            elif (dataset is not None) and (model is not None):
                if task_type is None:
                    raise ExperimentRunSettingError(
                        """
                        task_type must be provided when dataset and model are provided.
                        Please provide task_type="classification" or "regression".
                        """
                    )
                artifact, record = self._run_from_instance(
                    task_type=task_type,
                    dataset=dataset,
                    model=model,
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
            else:
                raise ExperimentRunSettingError("Either dataset and model or config must be provided.")
        except Exception as e:
            self.logger.error(f"Error occurred during the run: {e}")
            if add_results:
                self._run_backfill()
            raise e

        if add_results:
            self.exp_db.runs.append(record)  # type: ignore
            self.save_experiment()

            if config_source is not None:
                save_experiment_config_to_yaml(
                    config, Path(current_run_dirc) / DEFAULT_EXP_CONFIG_FILE, delete_source_path=True
                )
            else:
                # [TODO]: convert dataset and model instance to config and store it in the run directory
                pass

        return artifact, record

    def runs_to_dataframe(self) -> pd.DataFrame:
        """Convert the run data to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame of run data

        Raises:
            ExperimentNotInitializedError: if the experiment is not initialized
        """
        self._is_initialized()
        run_data = [run.model_dump() for run in self.exp_db.runs]  # type: ignore
        run_data = [
            {"run_id": run_record_dict["run_id"], **run_record_dict["evaluation"]} for run_record_dict in run_data
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
        self, logging_evaluation: dict[str, float], reproduction_evaluation: dict[str, float]
    ) -> None:
        """Validate the evaluation results of logging and reproduction.

        Args:
            logging_evaluation (dict[str, float]): evaluation result of logging
            reproduction_evaluation (dict[str, float]): evaluation result of reproduction

        Raises:
            ValueError: if the evaluation results are different
        """
        invalid_dict: dict[str, str] = {}
        for key, value in logging_evaluation.items():
            reproduction_value = reproduction_evaluation.get(key, None)
            if value != reproduction_value:
                invalid_dict[key] = f"{value} -> {reproduction_value}"

        if len(invalid_dict) > 0:
            raise ValueError(
                f"Evaluation results are different between logging and reproduction (invalid metrics: {invalid_dict})."
            )

    def reproduce(self, run_id: int, check_commit_id: bool = False) -> tuple[RunArtifact, RunRecord]:
        """Reproduce the target run_id model from config file.
        If the target run_id does not have a config file path, raise an error.
        Reoroduce method not supported for the run executed from the instance.

        Args:
            run_id (int): target run_id
            check_commit_id (bool, optional): whether to check the commit_id. Defaults to False.

        Returns:
            tuple[RunArtifact, RunRecord]: artifact and record of the reproduced run_id

        Raises:
            ReproductinoError: if the run_id does not have a config file path
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
                "run from instance mode not supported for reproduction."
            )

        config_path = Path(f"{self.experiment_dirc}/run_{run_id}/{DEFAULT_EXP_CONFIG_FILE}")
        reproduced_artifact, reproduced_result = self.run(config_source=config_path, add_results=False)

        logging_evaluation = run_record.evaluation
        reproduced_evaluation = reproduced_result.evaluation
        self._validate_evaluation(logging_evaluation, reproduced_evaluation)
        self.logger.info(f"Reproduce model is successful. Evaluation results are the same as run_id={run_id}.")

        return reproduced_artifact, reproduced_result

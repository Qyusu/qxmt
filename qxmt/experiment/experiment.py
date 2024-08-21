import json
import subprocess
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from qxmt.constants import DEFAULT_EXP_DB_FILE, DEFAULT_EXP_DIRC, DEFAULT_MODEL_NAME, TZ
from qxmt.datasets.builder import DatasetBuilder
from qxmt.datasets.schema import Dataset
from qxmt.evaluation.evaluation import Evaluation
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    InvalidFileExtensionError,
    JsonEncodingError,
)
from qxmt.experiment.schema import ExperimentDB, RunRecord
from qxmt.logger import set_default_logger
from qxmt.models.base import BaseModel
from qxmt.models.builder import ModelBuilder

LOGGER = set_default_logger(__name__)


class Experiment:
    def __init__(
        self,
        name: Optional[str] = None,
        desc: Optional[str] = None,
        root_experiment_dirc: Path = DEFAULT_EXP_DIRC,
        logger: Logger = LOGGER,
    ) -> None:
        self.name: Optional[str] = name
        self.desc: Optional[str] = desc
        self.current_run_id: int = 0
        self.root_experiment_dirc: Path = root_experiment_dirc
        self.experiment_dirc: Path
        self.exp_db: Optional[ExperimentDB] = None
        self.logger: Logger = logger

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

    @staticmethod
    def _get_commit_id(logger: Logger) -> str:
        """Get the commit ID of the current git repository.
        if the current directory is not a git repository, return an empty string.

        Returns:
            str: git commit ID
        """
        try:
            commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            return commit_id
        except subprocess.CalledProcessError as e:
            logger.warning("Error executing git command:", e)
            return ""

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
        self.experiment_dirc.mkdir(parents=True)

        self.exp_db = ExperimentDB(
            name=self.name,
            desc=self.desc,
            working_dirc=Path.cwd(),
            experiment_dirc=self.experiment_dirc,
            runs=[],
        )

        return self

    def load_experiment(self, exp_file: str | Path) -> "Experiment":
        """Load existing experiment data from a json file.

        Args:
            exp_file (str | Path): path to the experiment json file

        Raises:
            FileNotFoundError: if the experiment file does not exist

        Returns:
            Experiment: loaded experiment
        """
        if not Path(exp_file).exists():
            raise FileNotFoundError(f"{exp_file} does not exist.")

        self._check_json_extension(exp_file)
        with open(exp_file, "r") as json_file:
            exp_data = json.load(json_file)

        self.exp_db = ExperimentDB(**exp_data)
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

        self.current_run_id = len(self.exp_db.runs)

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

    def run_evaluation(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Run evaluation for the current run.

        Args:
            actual (np.ndarray): array of actual values
            predicted (np.ndarray): array of predicted values

        Returns:
            dict: evaluation result
        """
        evaluation = Evaluation(
            actual=actual,
            predicted=predicted,
        )
        evaluation.evaluate()

        return evaluation.to_dict()

    def _run_from_config(
        self,
        config_path: str | Path,
        commit_id: str,
        run_dirc: str | Path,
    ) -> tuple[BaseModel, RunRecord]:
        """Run the experiment from the config file.

        Args:
            config_path (str | Path): path to the config file
            commit_id (str): commit ID of the current git repository
            run_dirc (str | Path): path to the run directory

        Returns:
            tuple[BaseModel, RunRecord]: model and run record of the current run
        """
        # [TODO]: receive config instance
        with open(config_path, "r") as yml:
            config = yaml.safe_load(yml)

        # [TODO]: handle raw_preprocess_logic and transform_logic
        dataset = DatasetBuilder(
            config.get("dataset"), raw_preprocess_logic=tmp_raw_preprocess, transform_logic=tmp_transform
        ).build()
        model = ModelBuilder(device_config=config.get("device"), model_config=config.get("model")).build()
        save_model_path = run_dirc / config.get("save_model_path", DEFAULT_MODEL_NAME)

        model, record = self._run_from_instance(
            dataset,
            model,
            save_model_path=save_model_path,
            desc=config.get("description", ""),
            commit_id=commit_id,
            config_path=config_path,
        )

        return model, record

    def _run_from_instance(
        self,
        dataset: Dataset,
        model: BaseModel,
        save_model_path: str | Path,
        desc: str,
        commit_id: str,
        config_path: str | Path = "",
    ) -> tuple[BaseModel, RunRecord]:
        """Run the experiment from the dataset and model instance.

        Args:
            dataset (Dataset): dataset object
            model (BaseModel): model object
            save_model_path (str | Path): path to save the model
            desc (str, optional): description of the run.
            commit_id (str): commit ID of the current git repository
            config_path (str | Path, optional): path to the config file. Defaults to "".

        Returns:
            tuple[BaseModel, RunRecord]: model and run record of the current run
        """
        model.fit(dataset.X_train, dataset.y_train)
        model.save(save_model_path)
        predicted = model.predict(dataset.X_test)

        record = RunRecord(
            run_id=self.current_run_id,
            desc=desc,
            commit_id=commit_id,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            config_path=config_path,
            evaluation=self.run_evaluation(dataset.y_test, predicted),
        )

        return model, record

    def run(
        self,
        dataset: Optional[Dataset] = None,
        model: Optional[BaseModel] = None,
        config_path: Optional[str | Path] = None,
        desc: str = "",
        add_record: bool = True,
    ) -> tuple[BaseModel, RunRecord]:
        """Start a new run for the experiment.
        run() method can be called two ways:
        1. Provide dataset and model instance
            This method is directory provided dataset and model instance. It is easy to use but less flexible.
            This method "NOT" track the experiment settings.
        2. Provide config_path
            This method is provided the path to the config file. It is more flexible but requires a config file.

        Args:
            dataset (Dataset): dataset object
            model (BaseModel): model object
            config_path (str | Path, optional): path to the config file. Defaults to None.
            desc (str, optional): description of the run. Defaults to "".
            add_record (bool, optional): whether to add the run record to the experiment. Defaults to True.

        Returns:
            tuple[BaseModel, RunRecord]: model and run record of the current run

        Raises:
            ExperimentNotInitializedError: if the experiment is not initialized
        """
        self._is_initialized()
        current_run_dirc = self._run_setup()
        commit_id = self._get_commit_id(self.logger)

        if config_path is not None:
            model, record = self._run_from_config(config_path, commit_id, run_dirc=current_run_dirc)
        elif (dataset is not None) and (model is not None):
            save_model_path = current_run_dirc / DEFAULT_MODEL_NAME
            model, record = self._run_from_instance(dataset, model, save_model_path, desc, commit_id)
        else:
            raise ExperimentRunSettingError("Either dataset and model or config_path must be provided.")

        if add_record:
            self.exp_db.runs.append(record)  # type: ignore

        return model, record

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

    def reproduction(self, run_id: int) -> tuple[BaseModel, RunRecord]:
        """Reproduction of the target run.

        Args:
            run_id (int): target run_id

        Returns:
            tuple[BaseModel, RunRecord]: model and run record of the current run
        """
        self._is_initialized()
        run_record = self.get_run_record(self.exp_db.runs, run_id)  # type: ignore
        config_path = run_record.config_path
        reproduction_model, reproduction_result = self.run(config_path=config_path, add_record=False)

        logging_evaluation = run_record.evaluation
        reproduction_evaluation = reproduction_result.evaluation
        self._validate_evaluation(logging_evaluation, reproduction_evaluation)

        return reproduction_model, reproduction_result


# [TODO]: Load from config file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qxmt.datasets.builder import PROCESSCED_DATASET_TYPE, RAW_DATASET_TYPE


def tmp_raw_preprocess(X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
    y = np.array([int(label) for label in y])
    indices = np.where(np.isin(y, [0, 1]))[0]
    X, y = X[indices][:100], y[indices][:100]

    return X, y


def tmp_transform(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> PROCESSCED_DATASET_TYPE:
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    pca.fit(x_train_scaled)
    X_train_pca = pca.transform(x_train_scaled)
    X_test_pca = pca.transform(x_test_scaled)

    return X_train_pca, y_train, X_test_pca, y_test

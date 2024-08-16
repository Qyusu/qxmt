import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from qxmt.constants import DEFAULT_EXP_DB_FILE, DEFAULT_EXP_DIRC, DEFAULT_MODEL_NAME, TZ
from qxmt.datasets.schema import Dataset
from qxmt.evaluation.evaluation import Evaluation
from qxmt.exceptions import (
    ExperimentNotInitializedError,
    InvalidFileExtensionError,
    JsonEncodingError,
)
from qxmt.experiment.schema import ExperimentDB, RunRecord
from qxmt.models.base import BaseModel


class Experiment:
    def __init__(
        self,
        name: Optional[str] = None,
        desc: Optional[str] = None,
        root_experiment_dirc: Path = DEFAULT_EXP_DIRC,
    ) -> None:
        self.name: Optional[str] = name
        self.desc: Optional[str] = desc
        self.current_run_id: int = 0
        self.root_experiment_dirc: Path = root_experiment_dirc
        self.experiment_dirc: Path
        self.exp_db: Optional[ExperimentDB] = None

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
    def _get_commit_id() -> str:
        """Get the commit ID of the current git repository.
        if the current directory is not a git repository, return an empty string.

        Returns:
            str: git commit ID
        """
        try:
            commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            return commit_id
        except subprocess.CalledProcessError as e:
            print("Error executing git command:", e)
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
            print(f'Name is changed from "{self.exp_db.name}" to "{self.name}".')
        else:
            self.name = self.exp_db.name

        if (self.desc is not None) and (self.desc != self.exp_db.desc):
            self.exp_db.desc = self.desc
            print(f'Description is changed from "{self.exp_db.desc}" to "{self.desc}".')
        else:
            self.desc = self.exp_db.desc

        working_dirc = Path.cwd()
        if working_dirc != self.exp_db.working_dirc:
            print(f'Working directory is changed from "{self.exp_db.working_dirc}" to "{working_dirc}".')
            self.exp_db.working_dirc = working_dirc

        self.experiment_dirc = self.root_experiment_dirc / str(self.name)
        if self.experiment_dirc != self.exp_db.experiment_dirc:
            print(f'Experiment directory is changed from "{self.exp_db.experiment_dirc}" to "{self.experiment_dirc}".')
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
        """Setup for the current run."""
        self.current_run_id += 1
        current_run_dirc = self.experiment_dirc / f"run_{self.current_run_id}"
        current_run_dirc.mkdir(parents=True)

        return current_run_dirc

    def _run_evaluation(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Run evaluation for the current run.

        Args:
            actual (np.ndarray): array of actual values
            predicted (np.ndarray): array of predicted values

        Returns:
            Evaluation: evaluation result
        """
        evaluation = Evaluation(
            actual=actual,
            predicted=predicted,
        )
        evaluation.evaluate()

        return evaluation.to_dict()

    def run(self, dataset: Dataset, model: BaseModel, desc: str = "") -> None:
        """Start a new run for the experiment."""
        self._is_initialized()
        current_run_dirc = self._run_setup()
        commit_id = self._get_commit_id()

        model.fit(dataset.X_train, dataset.y_train)
        model.save(current_run_dirc / DEFAULT_MODEL_NAME)
        predicted = model.predict(dataset.X_test)

        current_run_record = RunRecord(
            run_id=self.current_run_id,
            desc=desc,
            commit_id=commit_id,
            execution_time=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z%z"),
            evaluation=self._run_evaluation(dataset.y_test, predicted),
        )

        self.exp_db.runs.append(current_run_record)  # type: ignore

    def runs_to_dataframe(self) -> pd.DataFrame:
        """Convert the run data to a pandas DataFrame."""
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

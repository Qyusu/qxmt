import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from qxmt.constants import DEFAULT_EXP_DB_FILE, DEFAULT_EXP_DIRC, TZ
from qxmt.datasets.schema import Dataset
from qxmt.evaluation.evaluation import Evaluation
from qxmt.exceptions import ExperimentNotInitializedError, JsonEncodingError
from qxmt.experiment.schema import ExperimentDB, RunRecord
from qxmt.models.base import BaseModel
from qxmt.utils import check_json_extension


class Experiment:
    def __init__(
        self,
        name: Optional[str] = None,
        desc: str = "",
        root_experiment_dirc: Path = DEFAULT_EXP_DIRC,
    ) -> None:
        self.name: str = name or self._generate_default_name()
        self.desc: str = desc
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

    def _init_db(self) -> None:
        """Initialize the experiment database.

        Raises:
            ValidationError: if the experiment database is not valid
        """
        self.exp_db = ExperimentDB(
            name=self.name,
            desc=self.desc,
            experiment_dirc=self.experiment_dirc,
            runs=[],
        )

    def init(self) -> "Experiment":
        """Initialize the experiment directory and DB.

        Returns:
            Experiment: initialized experiment
        """
        self.experiment_dirc = self.root_experiment_dirc / self.name
        self.experiment_dirc.mkdir(parents=True)
        self._init_db()

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

        check_json_extension(exp_file)
        with open(exp_file, "r") as json_file:
            exp_data = json.load(json_file)

        self.exp_db = ExperimentDB(**exp_data)
        self.name = self.exp_db.name
        self.desc = self.exp_db.desc
        self.experiment_dirc = self.root_experiment_dirc / self.name
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

        model.fit(dataset.x_train, dataset.y_train)
        model.save(current_run_dirc / "model.pkl")
        predicted = model.predict(dataset.x_test)

        current_run_record = RunRecord(
            run_id=self.current_run_id,
            desc=desc,
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
        check_json_extension(save_path)
        exp_data = json.loads(self.exp_db.model_dump_json())  # type: ignore
        with open(save_path, "w") as json_file:
            json.dump(exp_data, json_file, indent=4, default=custom_encoder)

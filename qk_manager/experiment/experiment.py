import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError

from qk_manager.constants import DEFAULT_EXP_DB_FILE, DEFAULT_EXP_DIRC
from qk_manager.evaluation.evaluation import Evaluation
from qk_manager.exceptions import JsonEncodingError
from qk_manager.experiment.schema import ExperimentDB, RunRecord
from qk_manager.utils import check_json_extension


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
        self.experiment_dirc = root_experiment_dirc / self.name
        self._init_db()
        self._create_experiment_dirc()

    @staticmethod
    def _generate_default_name() -> str:
        """Generate a default name for the experiment.
        Default name is the current date and time in the format of
        "YYYYMMDDHHMMSSffffff"

        Returns:
            str: generated default name
        """
        return datetime.now().strftime("%Y%m%d%H%M%S%f")

    def _create_experiment_dirc(self) -> None:
        """Create a empty directory for the experiment.

        Args:
            root_experiment_dirc (Path): root directory for the experiment

        Returns:
            Path: path to the created directory
        """
        self.experiment_dirc.mkdir(parents=True)

    def _init_db(self) -> None:
        """Initialize the experiment database.

        Raises:
            ValidationError: if the experiment database is not valid
        """
        try:
            self.exp_db = ExperimentDB(
                name=self.name,
                desc=self.desc,
                experiment_dirc=self.experiment_dirc,
                runs=[],
            )
        except ValidationError as e:
            print(e.json())
            raise e

    def run(self) -> None:
        """Start a new run for the experiment."""
        self.current_run_id += 1

        # [TODO]: replace this dummy data with actual data
        dummy_actual = np.random.randint(2, size=100)
        dummy_predicted = np.random.randint(2, size=100)

        current_run_record = RunRecord(
            run_id=self.current_run_id,
            desc="",  # [TODO]: add description
            evaluation=self._run_evaluation(dummy_actual, dummy_predicted),
        )

        self.exp_db.runs.append(current_run_record)

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

    def runs_to_dataframe(self) -> pd.DataFrame:
        """Convert the run data to a pandas DataFrame."""
        run_data = [run.model_dump() for run in self.exp_db.runs]
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

        def custom_encoder(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise JsonEncodingError(f"Object of type {type(obj).__name__} is not JSON serializable")

        save_path = self.experiment_dirc / exp_file
        check_json_extension(save_path)
        exp_data = json.loads(self.exp_db.model_dump_json())
        with open(save_path, "w") as json_file:
            json.dump(exp_data, json_file, indent=4, default=custom_encoder)

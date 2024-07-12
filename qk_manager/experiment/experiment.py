from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from qk_manager.constants import (
    DEFAULT_EXP_DB_FILE,
    DEFAULT_EXP_DIRC,
    DEFAULT_RUN_DB_FILE,
)
from qk_manager.experiment.schema import ExperimentDB, ExperimentRecord, RunRecord


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
        self.experiment_dirc = self._create_experiment_dirc(root_experiment_dirc)
        self._init_db()

    @staticmethod
    def _generate_default_name() -> str:
        """Generate a default name for the experiment.
        Default name is the current date and time in the format of
        "YYYYMMDDHHMMSSffffff"

        Returns:
            str: generated default name
        """
        return datetime.now().strftime("%Y%m%d%H%M%S%f")

    def _create_experiment_dirc(self, root_experiment_dirc: Path) -> Path:
        """Create a empty directory for the experiment.

        Args:
            root_experiment_dirc (Path): root directory for the experiment

        Returns:
            Path: path to the created directory
        """
        experiment_dirc = root_experiment_dirc / self.name
        experiment_dirc.mkdir(parents=True)
        return experiment_dirc

    def _init_db(self) -> None:
        """Initialize the experiment database.

        Raises:
            ValidationError: if the experiment database is not valid
        """
        try:
            self.exp_db = ExperimentDB(
                experiment_info=ExperimentRecord(
                    name=self.name,
                    desc=self.desc,
                    experiment_dirc=self.experiment_dirc,
                ),
                run_info=[],
            )
        except ValidationError as e:
            print(e.json())

    def run(self) -> None:
        """Start a new run for the experiment."""
        self.current_run_id += 1
        current_run_record = RunRecord(run_id=self.current_run_id)
        self.exp_db.run_info.append(current_run_record)

    def save_results(self) -> None:
        """Save the results of the experiment."""
        exp_db_df = pd.DataFrame([asdict(self.exp_db.experiment_info)])
        exp_db_df.to_csv(self.experiment_dirc / DEFAULT_EXP_DB_FILE, index=False)

        run_data = [asdict(run) for run in self.exp_db.run_info]
        run_db_df = pd.DataFrame(run_data)
        run_db_df.to_csv(self.experiment_dirc / DEFAULT_RUN_DB_FILE, index=False)

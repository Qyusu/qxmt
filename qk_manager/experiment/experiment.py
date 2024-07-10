from datetime import datetime
from pathlib import Path
from typing import Optional

from qk_manager.constants import DEFAULT_EXP_DIRC


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

    def run(self) -> None:
        raise NotImplementedError

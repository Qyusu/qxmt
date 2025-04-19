import json
import logging
import shutil
from pathlib import Path
from typing import Any

from qxmt.exceptions import InvalidFileExtensionError, JsonEncodingError
from qxmt.experiment.schema import ExperimentDB


class ExperimentRepository:
    """Repository class for handling experiment-related file system operations.

    This class is responsible for managing the persistence layer of experiments,
    including directory creation, file operations, and JSON serialization/deserialization.
    It separates side-effecting operations from the core business logic.

    Attributes:
        _logger (logging.Logger): Logger instance for logging operations.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes the ExperimentRepository.

        Args:
            logger (logging.Logger): Logger instance for logging operations.
        """
        self._logger = logger

    @staticmethod
    def check_json_extension(file_path: str | Path) -> None:
        """Validates if the given file has a .json extension.

        Args:
            file_path (str | Path): Path to the file to be checked.

        Raises:
            InvalidFileExtensionError: If the file does not have a .json extension.
        """
        if Path(file_path).suffix.lower() != ".json":
            raise InvalidFileExtensionError(f"File '{file_path}' does not have a '.json' extension.")

    def create_run_dir(self, experiment_dirc: str | Path, run_id: int) -> Path:
        """Creates a new run directory under the experiment directory.

        Args:
            experiment_dirc (str | Path): Path to the experiment directory.
            run_id (int): ID of the run to create directory for.

        Returns:
            Path: Path to the created run directory.

        Raises:
            Exception: If the run directory already exists or if there's an error
                during directory creation.
        """
        run_dirc = Path(experiment_dirc) / f"run_{run_id}"
        try:
            run_dirc.mkdir(parents=True)
        except FileExistsError:
            raise Exception(f"Run directory '{run_dirc}' already exists.")
        except Exception as e:
            self._logger.error(f"Error creating run directory: {e}")
            raise
        return run_dirc

    def remove_run_dir(self, experiment_dirc: str | Path, run_id: int) -> None:
        """Removes a run directory.

        Used for rollback or back-fill operations.

        Args:
            experiment_dirc (str | Path): Path to the experiment directory.
            run_id (int): ID of the run whose directory should be removed.
        """
        run_dirc = Path(experiment_dirc) / f"run_{run_id}"
        if run_dirc.exists():
            shutil.rmtree(run_dirc)

    def save(self, exp_db: ExperimentDB, save_path: str | Path) -> None:
        """Serializes experiment data into JSON format and saves it to disk.

        Args:
            exp_db (ExperimentDB): Experiment database object to be serialized.
            save_path (str | Path): Path where the JSON file should be saved.

        Raises:
            JsonEncodingError: If the object contains non-serializable data.
            InvalidFileExtensionError: If the save path does not have a .json extension.
        """

        def custom_encoder(obj: Any) -> str:
            """Custom JSON encoder for handling Path objects.

            Args:
                obj (Any): Object to be encoded.

            Returns:
                str: String representation of the object.

            Raises:
                JsonEncodingError: If the object type is not supported.
            """
            if isinstance(obj, Path):
                return str(obj)
            raise JsonEncodingError(f"Object of type {type(obj).__name__} is not JSON serializable")

        self.check_json_extension(save_path)
        # Convert pydantic model to python dict first to avoid Path issues
        exp_data = json.loads(exp_db.model_dump_json())

        with open(save_path, "w") as json_file:
            json.dump(exp_data, json_file, indent=4, default=custom_encoder)

    def load(self, exp_file_path: str | Path) -> ExperimentDB:
        """Loads experiment data from a JSON file.

        Args:
            exp_file_path (str | Path): Path to the JSON file containing experiment data.

        Returns:
            ExperimentDB: Loaded experiment database object.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            InvalidFileExtensionError: If the file does not have a .json extension.
        """
        exp_file_path = Path(exp_file_path)
        self.check_json_extension(exp_file_path)

        if not exp_file_path.exists():
            raise FileNotFoundError(f"{exp_file_path} does not exist.")

        with open(exp_file_path, "r") as json_file:
            exp_data = json.load(json_file)
        return ExperimentDB(**exp_data)

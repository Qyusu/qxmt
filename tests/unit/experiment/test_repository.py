from pathlib import Path
from unittest.mock import Mock

import pytest

from qxmt.exceptions import InvalidFileExtensionError
from qxmt.experiment.repository import ExperimentRepository
from qxmt.experiment.schema import ExperimentDB


class TestExperimentRepository:
    @pytest.fixture
    def repository(self) -> ExperimentRepository:
        return ExperimentRepository(logger=Mock())

    @pytest.fixture
    def experiment_db(self) -> ExperimentDB:
        return ExperimentDB(
            name="test_exp",
            desc="test experiment",
            working_dirc=Path.cwd(),
            experiment_dirc=Path.cwd() / "test_exp",
            runs=[],
        )

    def test_check_json_extension(self, repository: ExperimentRepository) -> None:
        # Valid JSON extension
        repository.check_json_extension("test.json")
        repository.check_json_extension(Path("test.json"))

        # Invalid JSON extension
        with pytest.raises(InvalidFileExtensionError):
            repository.check_json_extension("test.txt")
        with pytest.raises(InvalidFileExtensionError):
            repository.check_json_extension(Path("test.txt"))

    def test_create_run_dir(self, repository: ExperimentRepository, tmp_path: Path) -> None:
        # Success case
        run_dir = repository.create_run_dir(tmp_path, 1)
        assert run_dir.exists()
        assert run_dir.name == "run_1"

        # Directory already exists
        with pytest.raises(Exception, match="already exists"):
            repository.create_run_dir(tmp_path, 1)

    def test_remove_run_dir(self, repository: ExperimentRepository, tmp_path: Path) -> None:
        # Create a run directory first
        run_dir = tmp_path / "run_1"
        run_dir.mkdir()

        # Remove the directory
        repository.remove_run_dir(tmp_path, 1)
        assert not run_dir.exists()

        # Remove non-existent directory (should not raise error)
        repository.remove_run_dir(tmp_path, 2)

    def test_save(self, repository: ExperimentRepository, experiment_db: ExperimentDB, tmp_path: Path) -> None:
        # Success case
        save_path = tmp_path / "experiment.json"
        repository.save(experiment_db, save_path)
        assert save_path.exists()

        # Invalid file extension
        with pytest.raises(InvalidFileExtensionError):
            repository.save(experiment_db, tmp_path / "experiment.txt")

    def test_load(self, repository: ExperimentRepository, experiment_db: ExperimentDB, tmp_path: Path) -> None:
        # Save experiment data first
        save_path = tmp_path / "experiment.json"
        repository.save(experiment_db, save_path)

        # Success case
        loaded_db = repository.load(save_path)
        assert loaded_db.name == experiment_db.name
        assert loaded_db.desc == experiment_db.desc
        assert loaded_db.working_dirc == experiment_db.working_dirc
        assert loaded_db.experiment_dirc == experiment_db.experiment_dirc

        # File not found
        with pytest.raises(FileNotFoundError):
            repository.load(tmp_path / "non_existent.json")

        # Invalid file extension
        with pytest.raises(InvalidFileExtensionError):
            repository.load(tmp_path / "experiment.txt")

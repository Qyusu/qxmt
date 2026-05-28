import shutil
from pathlib import Path
from typing import Generator

import pytest


def _remove_qchem_dataset_cache() -> None:
    dataset_cache_dir = Path("datasets") / "qchem"
    shutil.rmtree(dataset_cache_dir, ignore_errors=True)

    dataset_root = dataset_cache_dir.parent
    if dataset_root.exists() and not any(dataset_root.iterdir()):
        dataset_root.rmdir()


@pytest.fixture(autouse=True)
def cleanup_qchem_dataset_cache() -> Generator[None, None, None]:
    _remove_qchem_dataset_cache()
    yield
    _remove_qchem_dataset_cache()

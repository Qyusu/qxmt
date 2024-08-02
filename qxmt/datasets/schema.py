from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pydantic.dataclasses import dataclass


class DatasetConfig(BaseModel):
    dataset_path: Path | str
    random_state: int
    test_size: float
    features: Optional[list[str]] = None

    class Config:
        arbitrary_types_allowed = True


class Dataset(BaseModel):
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    config: DatasetConfig

    class Config:
        arbitrary_types_allowed = True

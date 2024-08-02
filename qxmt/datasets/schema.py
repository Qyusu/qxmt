from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
from pydantic import BaseModel, PlainSerializer, PlainValidator


def validate(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    else:
        raise TypeError(f"Expected numpy array, got {type(v)}")


def serialize(v: np.ndarray) -> list[list[float]]:
    return v.tolist()


DataArray = Annotated[
    np.ndarray,
    PlainValidator(validate),
    PlainSerializer(serialize),
]


class DatasetConfig(BaseModel):
    dataset_path: Path | str
    random_state: int
    test_size: float
    features: Optional[list[str]] = None


class Dataset(BaseModel):
    X_train: DataArray
    y_train: DataArray
    X_test: DataArray
    y_test: DataArray
    config: DatasetConfig

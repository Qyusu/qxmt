from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, PlainSerializer, PlainValidator


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


class PathConfig(BaseModel):
    data: Path | str
    label: Path | str


class DatasetConfig(BaseModel):
    type: Literal["file", "generate"]
    path: PathConfig
    random_seed: int
    test_size: float = Field(ge=0.0, le=1.0)
    features: Optional[list[str]] = None


class Dataset(BaseModel):
    X_train: DataArray
    y_train: DataArray
    X_test: DataArray
    y_test: DataArray
    config: DatasetConfig

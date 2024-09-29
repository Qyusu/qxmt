from typing import Annotated, Any, Optional

import numpy as np
from pydantic import BaseModel, PlainSerializer, PlainValidator

from qxmt.configs import DatasetConfig


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


class Dataset(BaseModel):
    # model_config = ConfigDict(frozen=True, extra="forbid")

    X_train: DataArray
    y_train: DataArray
    X_val: Optional[DataArray]
    y_val: Optional[DataArray]
    X_test: DataArray
    y_test: DataArray
    config: DatasetConfig

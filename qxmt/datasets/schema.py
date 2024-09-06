from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, PlainSerializer, PlainValidator

from qxmt.configs import DatasetConfig
from qxmt.constants import MODULE_HOME


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

    def model_post_init(self, __context: dict[str, Any]) -> None:
        if not Path(self.data).is_absolute():
            self.data = MODULE_HOME / self.data

        if not Path(self.label).is_absolute():
            self.label = MODULE_HOME / self.label


class Dataset(BaseModel):
    X_train: DataArray
    y_train: DataArray
    X_test: DataArray
    y_test: DataArray
    config: DatasetConfig

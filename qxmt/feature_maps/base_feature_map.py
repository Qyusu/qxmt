from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMapParams:
    n_qubits: int
    reps: int
    rotaion_axis: Optional[str] = None


class BaseFeatureMap(ABC):
    def __init__(self, params: FeatureMapParams) -> None:
        self.params: FeatureMapParams = params

    def __call__(self, x: np.ndarray, n_qubits: int) -> None:
        self.feature_map(x, n_qubits)

    @abstractmethod
    def feature_map(self, x: np.ndarray, n_qubits: int) -> None:
        pass

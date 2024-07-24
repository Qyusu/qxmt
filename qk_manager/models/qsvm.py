import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from qk_manager.models.base_model import BaseKernelModel


class QSVM(BaseKernelModel):
    def __init__(self, **kwargs) -> None:
        self.model = SVC(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path: str | Path) -> None:
        pickle.load(open(path, "rb"))

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, params: dict) -> None:
        self.model.set_params(**params)

    def get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

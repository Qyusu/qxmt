import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseKernelModel


class QSVM(BaseKernelModel):
    def __init__(self, kernel: BaseKernel, **kwargs: dict) -> None:
        super().__init__(kernel)
        self.model = SVC(kernel=self.kernel.compute_matrix, **kwargs)  # type: ignore

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: dict) -> None:
        self.model.fit(X, y, **kwargs)  # type: ignore

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

import numpy as np
from sklearn.svm import SVC


class QSVM:
    def __init__(self) -> None:
        self.model = SVC()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

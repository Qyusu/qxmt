from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from qxmt.models import QSVM

SVM_PARAMS = [
    "kernel",
    "C",
    "degree",
    "gamma",
    "coef0",
    "tol",
    "cache_size",
    "class_weight",
    "verbose",
    "max_iter",
    "decision_function_shape",
    "break_ties",
    "random_state",
]


class TestQSVM:
    def test_save(self, build_qsvm: Callable, tmp_path: str) -> None:
        qsvm_model = build_qsvm()

        save_path = f"{tmp_path}/model.pkl"
        qsvm_model.save(save_path)
        assert Path(save_path).exists()

    def test_load(self, build_qsvm: Callable, tmp_path: str) -> None:
        qsvm_model = build_qsvm()

        save_path = f"{tmp_path}/model.pkl"
        qsvm_model.save(save_path)

        loaded_qsvm = QSVM(kernel=qsvm_model.kernel).load(save_path)
        for k, v in qsvm_model.get_params().items():
            if not callable(v):
                assert loaded_qsvm.get_params()[k] == v

    def test_fit(self, build_qsvm: Callable) -> None:
        qsvm_model = build_qsvm()

        with pytest.raises(AttributeError):
            qsvm_model.model.support_

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        qsvm_model.fit(X, y)
        qsvm_model.model.support_

    def test_predict(self, build_qsvm: Callable) -> None:
        qsvm_model = build_qsvm()

        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 1, 1, 0])
        qsvm_model.fit(X_train, y_train)

        X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_pred = qsvm_model.predict(X_test)
        assert y_pred.shape == (4,)
        assert all([0 <= y <= 1 for y in y_pred])

    def test_predict_proba(self, build_qsvm: Callable) -> None:
        qsvm_model_no_prob = build_qsvm(probability=False)
        qsvm_model_prob = build_qsvm(probability=True)

        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 1, 1, 0])
        qsvm_model_no_prob.fit(X_train, y_train)
        qsvm_model_prob.fit(X_train, y_train)

        X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        with pytest.raises(AttributeError):
            qsvm_model_no_prob.predict_proba(X_test)

        y_pred = qsvm_model_prob.predict_proba(X_test)
        assert y_pred.shape == (4, 2)
        assert all([0 <= y <= 1 for y in y_pred.flatten()])

    def test_get_params(self, build_qsvm: Callable) -> None:
        qsvm_model = build_qsvm()

        params = qsvm_model.get_params()
        for param in SVM_PARAMS:
            assert param in params

    def test_set_params(self, build_qsvm: Callable) -> None:
        qsvm_model = build_qsvm()

        params = qsvm_model.get_params()
        assert params["C"] != 100

        params["C"] = 100
        qsvm_model.set_params(params)
        assert qsvm_model.get_params()["C"] == 100

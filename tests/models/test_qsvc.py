from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from qxmt.models import QSVC

SVC_PARAMS = [
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


class TestQSVC:
    def test_cross_val_score(self, build_qsvc: Callable) -> None:
        qsvc_model = build_qsvc()

        X = np.random.rand(30, 2)
        y = np.random.randint(2, size=30)
        scores = qsvc_model.cross_val_score(X, y, cv=5)

        assert scores.shape == (5,)
        assert all([0 <= score <= 1 for score in scores])

    def test_fit(self, build_qsvc: Callable) -> None:
        qsvc_model = build_qsvc()

        with pytest.raises(AttributeError):
            qsvc_model.model.support_

        X = np.random.rand(10, 2)
        y = np.random.randint(2, size=10)
        qsvc_model.fit(X, y)
        qsvc_model.model.support_

    def test_predict(self, build_qsvc: Callable) -> None:
        qsvc_model = build_qsvc()

        X_train = np.random.rand(10, 2)
        y_train = np.random.randint(2, size=10)
        qsvc_model.fit(X_train, y_train)

        X_test = np.random.rand(10, 2)
        y_pred = qsvc_model.predict(X_test)
        assert y_pred.shape == (10,)
        assert all([0 <= y <= 1 for y in y_pred])

    def test_predict_proba(self, build_qsvc: Callable) -> None:
        qsvc_model_no_prob = build_qsvc(probability=False)
        qsvc_model_prob = build_qsvc(probability=True)

        X_train = np.random.rand(10, 2)
        y_train = np.random.randint(2, size=10)
        qsvc_model_no_prob.fit(X_train, y_train)
        qsvc_model_prob.fit(X_train, y_train)

        X_test = np.random.rand(10, 2)
        with pytest.raises(AttributeError):
            qsvc_model_no_prob.predict_proba(X_test)

        y_pred = qsvc_model_prob.predict_proba(X_test)
        assert y_pred.shape == (10, 2)
        assert all([0 <= y <= 1 for y in y_pred.flatten()])

    def test_save(self, build_qsvc: Callable, tmp_path: str) -> None:
        qsvc_model = build_qsvc()

        save_path = f"{tmp_path}/model.pkl"
        qsvc_model.save(save_path)
        assert Path(save_path).exists()

    def test_load(self, build_qsvc: Callable, tmp_path: str) -> None:
        qsvc_model = build_qsvc()

        save_path = f"{tmp_path}/model.pkl"
        qsvc_model.save(save_path)

        loaded_qsvm = QSVC(kernel=qsvc_model.kernel).load(save_path)
        for k, v in qsvc_model.get_params().items():
            if not callable(v):
                assert loaded_qsvm.get_params()[k] == v

    def test_get_params(self, build_qsvc: Callable) -> None:
        qsvc_model = build_qsvc()

        params = qsvc_model.get_params()
        for param in SVC_PARAMS:
            assert param in params

    def test_set_params(self, build_qsvc: Callable) -> None:
        qsvc_model = build_qsvc()

        params = qsvc_model.get_params()
        assert params["C"] != 100

        params["C"] = 100
        qsvc_model.set_params(params)
        assert qsvc_model.get_params()["C"] == 100

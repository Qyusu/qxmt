from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from qxmt.models.qkernels import QSVR

SVR_PARAMS = [
    "kernel",
    "C",
    "coef0",
    "degree",
    "epsilon",
    "gamma",
    "max_iter",
    "shrinking",
]


class TestQSVC:
    def test_cross_val_score(self, build_qsvr: Callable) -> None:
        qsvr_model = build_qsvr()

        X = np.random.rand(30, 2)
        y = np.random.randint(2, size=30)
        scores = qsvr_model.cross_val_score(X, y, cv=5)

        assert scores.shape == (5,)
        assert all([score <= 1 for score in scores])

    def test_fit(self, build_qsvr: Callable) -> None:
        qsvr_model = build_qsvr()

        with pytest.raises(AttributeError):
            qsvr_model.support_

        X = np.random.rand(10, 2)
        y = np.random.randint(2, size=10)
        qsvr_model.fit(X, y)
        qsvr_model.support_

    def test_predict(self, build_qsvr: Callable) -> None:
        qsvr_model = build_qsvr()

        X_train = np.random.rand(10, 2)
        y_train = np.random.randint(2, size=10)
        qsvr_model.fit(X_train, y_train)

        X_test = np.random.rand(10, 2)
        y_pred = qsvr_model.predict(X_test)
        assert y_pred.shape == (10,)

    def test_save(self, build_qsvr: Callable, tmp_path: str) -> None:
        qsvr_model = build_qsvr()

        save_path = f"{tmp_path}/model.pkl"
        qsvr_model.save(save_path)
        assert Path(save_path).exists()

    def test_load(self, build_qsvr: Callable, tmp_path: str) -> None:
        qsvr_model = build_qsvr()

        save_path = f"{tmp_path}/model.pkl"
        qsvr_model.save(save_path)

        loaded_qsvm = QSVR(kernel=qsvr_model.kernel).load(save_path)
        for k, v in qsvr_model.get_params().items():
            if not callable(v):
                assert loaded_qsvm.get_params()[k] == v

    def test_get_params(self, build_qsvr: Callable) -> None:
        qsvr_model = build_qsvr()

        params = qsvr_model.get_params()
        for param in SVR_PARAMS:
            assert param in params

    def test_set_params(self, build_qsvr: Callable) -> None:
        qsvr_model = build_qsvr()

        params = qsvr_model.get_params()
        assert params["C"] != 100

        params["C"] = 100
        qsvr_model.set_params(params)
        assert qsvr_model.get_params()["C"] == 100

from pathlib import Path
from typing import Callable

import numpy as np
from pytest_mock import MockFixture

from qxmt.models import QRiggeRegressor

KernelRigge_PARAMS = [
    "alpha",
    "kernel",
    "gamma",
    "degree",
    "coef0",
    "kernel_params",
]


class TestQRiggeRegressor:
    def test_fit(self, build_qrigge: Callable, mocker: MockFixture) -> None:
        qrigge_model = build_qrigge()
        mock_kernel_matrix = np.eye(5)
        mocker.patch.object(qrigge_model.kernel, "compute_matrix", return_value=mock_kernel_matrix)

        assert qrigge_model.fit_X is None

        X = np.random.rand(5, 2)
        y = np.random.rand(5, 1)
        qrigge_model.fit(X, y)
        assert qrigge_model.fit_X is not None

    def test_predict(self, build_qrigge: Callable, mocker: MockFixture) -> None:
        qrigge_model = build_qrigge()
        mock_kernel_matrix = np.eye(5)
        mocker.patch.object(qrigge_model.kernel, "compute_matrix", return_value=mock_kernel_matrix)

        X_train = np.random.rand(5, 2)
        y_train = np.random.rand(5, 1)
        qrigge_model.fit(X_train, y_train)

        X_test = np.random.rand(5, 2)
        y_pred = qrigge_model.predict(X_test)
        assert y_pred.shape == (5, 1)

    def test_score(self, build_qrigge: Callable, mocker: MockFixture) -> None:
        qrigge_model = build_qrigge()
        mock_kernel_matrix = np.eye(5)
        mocker.patch.object(qrigge_model.kernel, "compute_matrix", return_value=mock_kernel_matrix)

        X_train = np.random.rand(5, 2)
        y_train = np.random.rand(5, 1)
        qrigge_model.fit(X_train, y_train)

        X_test = np.random.rand(5, 2)
        y_test = np.random.rand(5, 1)
        score = qrigge_model.score(X_test, y_test)
        assert isinstance(score, float)

    def test_save(self, build_qrigge: Callable, tmp_path: str) -> None:
        qrigge_model = build_qrigge()

        save_path = f"{tmp_path}/model.pkl"
        qrigge_model.save(save_path)
        assert Path(save_path).exists()

    def test_load(self, build_qrigge: Callable, tmp_path: str) -> None:
        qrigge_model = build_qrigge()

        save_path = f"{tmp_path}/model.pkl"
        qrigge_model.save(save_path)

        loaded_qrigge = QRiggeRegressor(kernel=qrigge_model.kernel).load(save_path)
        for k, v in qrigge_model.get_params().items():
            if not callable(v):
                assert getattr(loaded_qrigge, k) == v

    def test_get_params(self, build_qrigge: Callable) -> None:
        qrigge_model = build_qrigge()
        params = qrigge_model.get_params()
        for param in KernelRigge_PARAMS:
            assert param in params

    def test_set_params(self, build_qrigge: Callable) -> None:
        qrigge_model = build_qrigge()

        params = qrigge_model.get_params()
        assert params["alpha"] != 100

        params["alpha"] = 100
        qrigge_model.set_params(params)
        assert qrigge_model.get_params()["alpha"] == 100

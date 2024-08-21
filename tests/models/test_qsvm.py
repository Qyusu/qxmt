from pathlib import Path

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
    def test_save(self, qsvm_model: QSVM, tmp_path: str) -> None:
        save_path = f"{tmp_path}/model.pkl"
        qsvm_model.save(save_path)
        assert Path(save_path).exists()

    def test_load(self, qsvm_model: QSVM, tmp_path: str) -> None:
        save_path = f"{tmp_path}/model.pkl"
        qsvm_model.save(save_path)

        loaded_qsvm = QSVM(kernel=qsvm_model.kernel).load(save_path)
        for k, v in qsvm_model.get_params().items():
            if not callable(v):
                assert loaded_qsvm.get_params()[k] == v

    def test_get_params(self, qsvm_model: QSVM) -> None:
        params = qsvm_model.get_params()
        for param in SVM_PARAMS:
            assert param in params

    def test_set_params(self, qsvm_model: QSVM) -> None:
        params = qsvm_model.get_params()
        assert params["C"] != 100
        params["C"] = 100
        qsvm_model.set_params(params)
        assert qsvm_model.get_params()["C"] == 100

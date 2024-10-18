import numpy as np

from qxmt.datasets.generate import GeneratedDataLoader


class TestGeneratedDataLoader:
    def test_load_classification_liner(self) -> None:
        loader = GeneratedDataLoader(
            task_type="classification",
            generate_method="linear",
            random_seed=0,
            params={"n_samples": 100, "n_features": 2, "n_classes": 2},
        )
        X, y = loader.load()
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert len(np.unique(y)) <= 2

    def test_load_regression_liner(self) -> None:
        loader = GeneratedDataLoader(
            task_type="regression",
            generate_method="linear",
            random_seed=0,
            params={"n_samples": 5, "n_features": 1},
        )
        X, y = loader.load()
        assert X.shape == (5, 1)
        assert y.shape == (5,)

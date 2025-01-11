import numpy as np

from qxmt.datasets.transform.reduction_by_pca import dimension_reduction_by_pca


class TestDimensionReduction:
    def test_dimension_reduction_by_pca(self) -> None:
        X_train = np.array([[-3, 2], [6, -4], [5, 6]])
        y_train = np.array([0, 1, 0])
        X_val = np.array([[2, -1], [4, 5]])
        y_val = np.array([1, 0])
        X_test = np.array([[3, 4], [5, 6]])
        y_test = np.array([0, 1])

        # Not using normalization
        X_train_pca, y_train_pca, X_val_pca, y_val_pca, X_test_pca, y_test_pca = dimension_reduction_by_pca(
            X_train, y_train, X_val, y_val, X_test, y_test, n_components=2, random_seed=42, normalize=False
        )

        assert X_train_pca.shape == (3, 2)
        assert X_val_pca is not None and X_val_pca.shape == (2, 2)
        assert X_test_pca.shape == (2, 2)
        assert np.array_equal(y_train_pca, y_train)
        assert y_val_pca is not None and np.array_equal(y_val_pca, y_val)
        assert np.array_equal(y_test_pca, y_test)

        # Using normalization
        X_train_pca, y_train_pca, X_val_pca, y_val_pca, X_test_pca, y_test_pca = dimension_reduction_by_pca(
            X_train, y_train, X_val, y_val, X_test, y_test, n_components=2, random_seed=42, normalize=True
        )

        assert X_train_pca.shape == (3, 2)
        assert X_val_pca is not None and X_val_pca.shape == (2, 2)
        assert X_test_pca.shape == (2, 2)
        assert np.array_equal(y_train_pca, y_train)
        assert y_val_pca is not None and np.array_equal(y_val_pca, y_val)
        assert np.array_equal(y_test_pca, y_test)
        tolerance = 1e-6
        assert np.all((X_train_pca >= 0.0 - tolerance) & (X_train_pca <= 1.0 + tolerance))
        assert np.all((X_val_pca >= 0.0 - tolerance) & (X_val_pca <= 1.0 + tolerance))
        assert np.all((X_test_pca >= 0.0 - tolerance) & (X_test_pca <= 1.0 + tolerance))

import numpy as np

from qxmt.datasets.transform.normalizer import normalization


class TestNormalization:
    def test_normalization_by_standard(self) -> None:
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        X_val = np.array([[2, 3], [4, 5]])
        y_val = np.array([1, 0])
        X_test = np.array([[3, 4], [5, 6]])
        y_test = np.array([0, 1])

        # using validation set
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = normalization(
            X_train, y_train, X_val, y_val, X_test, y_test, scaler_type="StandardScaler"
        )
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled is not None and X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape
        assert np.array_equal(y_train_scaled, y_train)
        assert y_val_scaled is not None and np.array_equal(y_val_scaled, y_val)
        assert np.array_equal(y_test_scaled, y_test)

        # Not using validation set
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = normalization(
            X_train, y_train, None, None, X_test, y_test, scaler_type="StandardScaler"
        )

        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled is None
        assert X_test_scaled.shape == X_test.shape
        assert np.array_equal(y_train_scaled, y_train)
        assert y_val_scaled is None
        assert np.array_equal(y_test_scaled, y_test)

    def test_normalization_by_minmax(self) -> None:
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        X_val = np.array([[2, 3], [4, 5]])
        y_val = np.array([1, 0])
        X_test = np.array([[3, 4], [5, 6]])
        y_test = np.array([0, 1])

        # using validation set
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = normalization(
            X_train, y_train, X_val, y_val, X_test, y_test, scaler_type="MinMaxScaler"
        )

        assert np.min(X_train_scaled) >= 0 and np.max(X_train_scaled) <= 1
        assert X_val_scaled is not None and np.min(X_val_scaled) >= 0 and np.max(X_val_scaled) <= 1
        assert np.min(X_test_scaled) >= 0 and np.max(X_test_scaled) <= 1
        assert np.array_equal(y_train_scaled, y_train)
        assert y_val_scaled is not None and np.array_equal(y_val_scaled, y_val)
        assert np.array_equal(y_test_scaled, y_test)

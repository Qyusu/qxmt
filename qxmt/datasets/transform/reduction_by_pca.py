import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qxmt.datasets.builder import PROCESSCED_DATASET_TYPE, RAW_DATASET_TYPE


def dimension_reduction_by_pca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> PROCESSCED_DATASET_TYPE:
    """Dimension reduction by PCA

    Args:
        X_train (np.ndarray): numpy array of training data
        y_train (np.ndarray): numpy array of training label
        X_test (np.ndarray): numpy array of testing data
        y_test (np.ndarray): numpy array of testing label
        n_components (int): number of components to keep

    Returns:
        PROCESSCED_DATASET_TYPE: tuple of processed dataset
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    pca.fit(x_train_scaled)
    X_train_pca = pca.transform(x_train_scaled)
    X_test_pca = pca.transform(x_test_scaled)

    return X_train_pca, y_train, X_test_pca, y_test

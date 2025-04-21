from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qxmt.types import PROCESSCED_DATASET_TYPE


def dimension_reduction_by_pca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
    random_seed: Optional[int] = None,
    normalize: bool = False,
) -> PROCESSCED_DATASET_TYPE:
    """Dimension reduction by PCA

    Args:
        X_train (np.ndarray): numpy array of training data
        y_train (np.ndarray): numpy array of training label
        X_val (Optional[np.ndarray]): numpy array of validation data. None if validation set is not used
        y_val (Optional[np.ndarray]): numpy array of validation data. None if validation set is not used
        X_test (np.ndarray): numpy array of testing data
        y_test (np.ndarray): numpy array of testing label
        n_components (int): number of components to keep
        random_seed (Optional[int]): random seed for PCA
        normalize (bool): whether to normalize the output after PCA

    Returns:
        PROCESSCED_DATASET_TYPE: tuple of processed dataset
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled) if X_val_scaled is not None else None
    X_test_pca = pca.transform(X_test_scaled)

    if normalize:
        min_max_scaler = MinMaxScaler()
        X_train_pca = min_max_scaler.fit_transform(X_train_pca)
        X_val_pca = min_max_scaler.transform(X_val_pca) if X_val_pca is not None else None
        X_test_pca = min_max_scaler.transform(X_test_pca)

    return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test

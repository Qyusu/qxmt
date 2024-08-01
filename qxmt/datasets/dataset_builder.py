from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split

from qxmt.datasets.schema import Dataset, DatasetConfig

RAW_DATASET_TYPE = tuple[np.ndarray, np.ndarray]
PROCESSCED_DATASET_TYPE = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DatasetBuilder(ABC):
    def __init__(self, config: dict) -> None:
        self.config: DatasetConfig = DatasetConfig(**config)

    @abstractmethod
    def load(self) -> RAW_DATASET_TYPE:
        """Load the dataset from the path defined in config.

        Returns:
            RAW_DATASET_TYPE: features and labels of the dataset
        """
        pass

    def split(self, X: np.ndarray, y: np.ndarray) -> PROCESSCED_DATASET_TYPE:
        """Split the dataset into train and test sets.
        Test set size is defined in the config.

        Args:
            X (np.ndarray): raw features of the dataset
            y (np.ndarray): raw labels of the dataset

        Returns:
            PROCESSCED_DATASET_TYPE: train and test split of dataset (features and labels)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        return X_train, y_train, X_test, y_test

    @abstractmethod
    def transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> PROCESSCED_DATASET_TYPE:
        """Transform and preprocess the dataset.

        Args:
            X_train (np.ndarray): raw features of the training data
            y_train (np.ndarray): raw labels of the training data
            X_test (np.ndarray): raw features of the test data
            y_test (np.ndarray): raw labels of the test data

        Returns:
            PROCESSCED_DATASET_TYPE: transformed train and test split of dataset (features and labels)
        """
        pass

    def build(self) -> Dataset:
        X, y = self.load()
        X_train, y_train, X_test, y_test = self.split(X, y)
        X_train_trs, y_train_trs, X_test_trs, y_test_trs = self.transform(X_train, y_train, X_test, y_test)

        return Dataset(
            X_train=X_train_trs,
            y_train=y_train_trs,
            X_test=X_test_trs,
            y_test=y_test_trs,
            config=self.config,
        )

    def visualize(self) -> None:
        pass

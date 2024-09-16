import inspect
from logging import Logger
from typing import Callable, Optional, get_type_hints

import numpy as np
from sklearn.model_selection import train_test_split

from qxmt.configs import ExperimentConfig
from qxmt.datasets.dummy import generate_linear_separable_data
from qxmt.datasets.schema import Dataset
from qxmt.logger import set_default_logger
from qxmt.utils import load_object_from_yaml

RAW_DATA_TYPE = np.ndarray
RAW_LABEL_TYPE = np.ndarray
RAW_DATASET_TYPE = tuple[RAW_DATA_TYPE, RAW_LABEL_TYPE]
PROCESSCED_DATASET_TYPE = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

LOGGER = set_default_logger(__name__)


class DatasetBuilder:
    def __init__(self, config: ExperimentConfig, logger: Logger = LOGGER) -> None:
        self.config: ExperimentConfig = config
        self.logger: Logger = logger

        if self.config.dataset.raw_preprocess_logic is not None:
            raw_preprocess_logic = load_object_from_yaml(self.config.dataset.raw_preprocess_logic)
            self._validate_raw_preprocess_logic(raw_preprocess_logic, self.logger)
            self.custom_raw_preprocess: Optional[Callable] = raw_preprocess_logic
        else:
            self.custom_raw_preprocess = None

        if self.config.dataset.transform_logic is not None:
            transform_logic = load_object_from_yaml(self.config.dataset.transform_logic)
            self._validate_transform_logic(transform_logic, self.logger)
            self.custom_transform: Optional[Callable] = transform_logic
        else:
            self.custom_transform = None

    @staticmethod
    def _validate_raw_preprocess_logic(raw_preprocess_logic: Callable, logger: Logger) -> None:
        """Validate the custom raw preprocess function.

        Args:
            raw_preprocess_logic (Callable): custom raw preprocess function
            logger (Logger): logger for output messages

        Raises:
            ValueError: argment lenght of the custom raw preprocess function is less than 2
            ValueError: return type of the custom raw preprocess function is not a tuple of numpy arrays
            ValueError: argument type of the custom raw preprocess function is not numpy array
        """
        type_hint_dict = get_type_hints(raw_preprocess_logic)
        parameter_dict = inspect.signature(raw_preprocess_logic).parameters

        # check argment length. -1 means return type
        if len(type_hint_dict) - 1 != len(parameter_dict):
            logger.warning(
                "All arguments of the custom raw preprocess function assigned to the type hint."
                "Input and return type validation will be skipped."
            )
            return
        elif len(type_hint_dict) - 1 < 2:
            raise ValueError("The custom raw preprocess function must have at least 2 arguments (X, y).")

        # check argument type and return type
        for arg_name, arg_type in type_hint_dict.items():
            if (arg_name == "return") and (arg_type != RAW_DATASET_TYPE):
                raise ValueError(
                    "The return type of the custom raw preprocess function must be a tuple of numpy arrays."
                )
            # [TODO]: Handle anthor data types
            # elif (arg_name != "return") and (arg_type != RAW_DATA_TYPE):
            #     raise ValueError(f'The arguments of the custom raw preprocess function must be "{RAW_DATA_TYPE}".')

    @staticmethod
    def _validate_transform_logic(transform_logic: Callable, logger: Logger) -> None:
        """Validate the custom transform function.

        Args:
            transform_logic (Callable): custom transform function
            logger (Logger): logger for output messages

        Raises:
            ValueError: argment lenght of the custom transform function is less than 4
            ValueError: return type of the custom transform function is not a tuple of numpy arrays
            ValueError: argument type of the custom transform function is not numpy array

        """
        type_hint_dict = get_type_hints(transform_logic)
        parameter_dict = inspect.signature(transform_logic).parameters

        # check argment length. -1 means return type
        if len(type_hint_dict) - 1 != len(parameter_dict):
            logger.warning(
                "All arguments of the custom raw preprocess function assigned to the type hint."
                "Input and return type validation will be skipped."
            )
            return
        elif len(type_hint_dict) - 1 < 4:
            raise ValueError(
                "The custom transform function must have at least 4 arguments (X_train, y_train, X_test, y_test)."
            )

        # check argument type and return type
        for arg_name, arg_type in type_hint_dict.items():
            if (arg_name == "return") and (arg_type != PROCESSCED_DATASET_TYPE):
                raise ValueError("The return type of the custom transform function must be a tuple of numpy arrays.")
            # # [TODO]: Handle athor data types
            # elif (arg_name != "return") and (arg_type != RAW_DATA_TYPE):
            #     raise ValueError(f'The arguments of the custom transform function must be "{RAW_DATA_TYPE}".')

    def load(self) -> RAW_DATASET_TYPE:
        """Load the dataset from the path defined in config.

        Returns:
            RAW_DATASET_TYPE: features and labels of the dataset
        """
        if (self.config.dataset.type == "file") and (self.config.dataset.path is not None):
            # [TODO]: Implement other file formats
            X = np.load(self.config.dataset.path.data, allow_pickle=True)
            y = np.load(self.config.dataset.path.label, allow_pickle=True)
        elif self.config.dataset.type == "generate":
            # [TODO]: Implement other dataset generation methods
            X, y = generate_linear_separable_data()
        else:
            raise ValueError(f"Invalid dataset type: {self.config.dataset.type}")

        return X, y

    def default_raw_preprocess(self, X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
        """Default raw preprocess method. This method does not apply any preprocess.

        Args:
            X (np.ndarray): raw features of the dataset
            y (np.ndarray): raw labels of the dataset

        Returns:
            RAW_DATASET_TYPE: raw features and labels of the dataset
        """
        return X, y

    def raw_preprocess(self, X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
        """Preprocess the raw dataset. This process executes before splitting the dataset.
        ex) filtering, data sampling, etc.

        Args:
            X (np.ndarray): raw features of the dataset
            y (np.ndarray): raw labels of the dataset

        Returns:
            RAW_DATASET_TYPE: preprocessed features and labels of the dataset
        """
        if self.custom_raw_preprocess is not None:
            return self.custom_raw_preprocess(X, y)
        else:
            return self.default_raw_preprocess(X, y)

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
            X, y, test_size=self.config.dataset.test_size, random_state=self.config.dataset.random_seed
        )

        return X_train, y_train, X_test, y_test

    def default_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> PROCESSCED_DATASET_TYPE:
        """Default transform method. This method does not apply any transformation.

        Args:
            X_train (np.ndarray): raw features of the training data
            y_train (np.ndarray): raw labels of the training data
            X_test (np.ndarray): raw features of the test data
            y_test (np.ndarray): raw labels of the test data

        Returns:
            PROCESSCED_DATASET_TYPE: train and test split of dataset (features and labels)
        """
        return X_train, y_train, X_test, y_test

    def transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> PROCESSCED_DATASET_TYPE:
        """Transform the dataset.
        ex) feature scaling, dimension reduction, etc.

        Args:
            X_train (np.ndarray): raw features of the training data
            y_train (np.ndarray): raw labels of the training data
            X_test (np.ndarray): raw features of the test data
            y_test (np.ndarray): raw labels of the test data

        Returns:
            PROCESSCED_DATASET_TYPE: transformed train and test split of dataset (features and labels)
        """
        if self.custom_transform is not None:
            return self.custom_transform(X_train, y_train, X_test, y_test)
        else:
            return self.default_transform(X_train, y_train, X_test, y_test)

    def build(self) -> Dataset:
        X, y = self.load()
        X, y = self.raw_preprocess(X, y)
        X_train, y_train, X_test, y_test = self.split(X, y)
        X_train_trs, y_train_trs, X_test_trs, y_test_trs = self.transform(X_train, y_train, X_test, y_test)

        return Dataset(
            X_train=X_train_trs,
            y_train=y_train_trs,
            X_test=X_test_trs,
            y_test=y_test_trs,
            config=self.config.dataset,
        )

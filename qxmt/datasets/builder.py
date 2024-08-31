from typing import Callable, Optional, get_type_hints

import numpy as np
from sklearn.model_selection import train_test_split

from qxmt.datasets.dummy import generate_linear_separable_data
from qxmt.datasets.schema import Dataset, DatasetConfig
from qxmt.exceptions import InvalidConfigError
from qxmt.utils import extract_function_from_yaml

RAW_DATA_TYPE = np.ndarray
RAW_LABEL_TYPE = np.ndarray
RAW_DATASET_TYPE = tuple[RAW_DATA_TYPE, RAW_LABEL_TYPE]
PROCESSCED_DATASET_TYPE = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DatasetBuilder:
    def __init__(self, raw_config: dict) -> None:
        self.raw_config: dict = raw_config
        self.config: DatasetConfig = self._get_dataset_config()

        if self.config.raw_preprocess_logic is not None:
            raw_preprocess_logic = extract_function_from_yaml(self.config.raw_preprocess_logic)
            self._validate_raw_preprocess_logic(raw_preprocess_logic)
            self.custom_raw_preprocess: Optional[Callable] = raw_preprocess_logic
        else:
            self.custom_raw_preprocess = None

        if self.config.transform_logic is not None:
            transform_logic = extract_function_from_yaml(self.config.transform_logic)
            self._validate_transform_logic(transform_logic)
            self.custom_transform: Optional[Callable] = transform_logic
        else:
            self.custom_transform = None

    def _get_dataset_config(self, key: str = "dataset") -> DatasetConfig:
        """Get dataset configurations.

        Args:
            key (str, optional): key for device configuration. Defaults to "dataset".

        Raises:
            InvalidConfigError: key is not in the configuration file.
        """
        if key not in self.raw_config:
            raise InvalidConfigError(f"Key '{key}' is not in the configuration file.")

        return DatasetConfig(**self.raw_config[key])

    @staticmethod
    def _validate_raw_preprocess_logic(raw_preprocess_logic: Callable) -> None:
        """Validate the custom raw preprocess function.
        [TODO]: Handle the case when type hint is not defined.

        Args:
            raw_preprocess_logic (Callable): custom raw preprocess function

        Raises:
            ValueError: argment lenght of the custom raw preprocess function is not 2
            ValueError: return type of the custom raw preprocess function is not a tuple of numpy arrays
            ValueError: argument type of the custom raw preprocess function is not numpy array
        """
        type_hint_dict = get_type_hints(raw_preprocess_logic)
        # check argment length. -1 means return type
        if len(type_hint_dict) - 1 != 2:
            raise ValueError("The custom raw preprocess function must have exactly 2 arguments (X, y).")

        # check argument type and return type
        for arg_name, arg_type in type_hint_dict.items():
            if (arg_name == "return") and (arg_type != RAW_DATASET_TYPE):
                raise ValueError(
                    "The return type of the custom raw preprocess function must be a tuple of numpy arrays."
                )
            # [TODO]: Handle athor data types
            elif (arg_name != "return") and (arg_type != RAW_DATA_TYPE):
                raise ValueError(f'The arguments of the custom raw preprocess function must be "{RAW_DATA_TYPE}".')

    @staticmethod
    def _validate_transform_logic(transform_logic: Callable) -> None:
        """Validate the custom transform function.
        [TODO]: Handle the case when type hint is not defined.

        Args:
            transform_logic (Callable): custom transform function

        Raises:
            ValueError: argment lenght of the custom transform function is not 4
            ValueError: return type of the custom transform function is not a tuple of numpy arrays
            ValueError: argument type of the custom transform function is not numpy array

        """
        type_hint_dict = get_type_hints(transform_logic)
        # check argment length. -1 means return type
        if len(type_hint_dict) - 1 != 4:
            raise ValueError(
                "The custom transform function must have exactly 4 arguments (X_train, y_train, X_test, y_test)."
            )

        # check argument type and return type
        for arg_name, arg_type in type_hint_dict.items():
            if (arg_name == "return") and (arg_type != PROCESSCED_DATASET_TYPE):
                raise ValueError("The return type of the custom transform function must be a tuple of numpy arrays.")
            # [TODO]: Handle athor data types
            elif (arg_name != "return") and (arg_type != RAW_DATA_TYPE):
                raise ValueError(f'The arguments of the custom transform function must be "{RAW_DATA_TYPE}".')

    def load(self) -> RAW_DATASET_TYPE:
        """Load the dataset from the path defined in config.

        Returns:
            RAW_DATASET_TYPE: features and labels of the dataset
        """
        if self.config.type == "file":
            # [TODO]: Implement other file formats
            X = np.load(self.config.path.data, allow_pickle=True)
            y = np.load(self.config.path.label, allow_pickle=True)
        elif self.config.type == "generate":
            # [TODO]: Implement other dataset generation methods
            X, y = generate_linear_separable_data()
        else:
            raise ValueError(f"Invalid dataset type: {self.config.type}")

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
            X, y, test_size=self.config.test_size, random_state=self.config.random_seed
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
            config=self.config,
        )

from typing import Optional

import numpy as np
import pytest
from pytest_mock import MockFixture

from qxmt.configs import DatasetConfig, ExperimentConfig
from qxmt.datasets import DatasetBuilder

RAW_DATA_TYPE = np.ndarray
RAW_LABEL_TYPE = np.ndarray
RAW_DATASET_TYPE = tuple[RAW_DATA_TYPE, RAW_LABEL_TYPE]
PROCESSCED_DATASET_TYPE = tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray
]


class TestValidationPreprocessLogic:
    def test__validate_raw_preprocess_logic_valid(self, mocker: MockFixture) -> None:
        def valid_raw_preprocess_logic(X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
            return X, y

        mock_logger = mocker.MagicMock()
        DatasetBuilder._validate_raw_preprocess_logic(
            valid_raw_preprocess_logic,
            logger=mock_logger,  # type: ignore
        )

    def test__validate_raw_preprocess_logic_valid_no_typehint(self, mocker: MockFixture) -> None:
        def valid_raw_preprocess_logic_no_typehint(X, y):  # type: ignore
            return X, y

        mock_logger = mocker.MagicMock()
        DatasetBuilder._validate_raw_preprocess_logic(
            valid_raw_preprocess_logic_no_typehint,
            logger=mock_logger,  # type: ignore
        )
        mock_logger.warning.assert_called_with(
            "All arguments of the custom raw preprocess function assigned to the type hint."
            "Input and return type validation will be skipped."
        )

    def test__validate_raw_preprocess_logic_invalid(self, mocker: MockFixture) -> None:
        def invalid_args_raw_preprocess_logic(X: np.ndarray) -> RAW_DATASET_TYPE:
            return X, X

        mock_logger = mocker.MagicMock()
        with pytest.raises(ValueError):
            DatasetBuilder._validate_raw_preprocess_logic(
                invalid_args_raw_preprocess_logic,
                logger=mock_logger,  # type: ignore
            )

        def invalid_return_raw_preprocess_logic(
            X: np.ndarray, y: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return X, y, y

        with pytest.raises(ValueError):
            DatasetBuilder._validate_raw_preprocess_logic(
                invalid_return_raw_preprocess_logic,
                logger=mock_logger,  # type: ignore
            )

        # def invalid_arg_type_raw_preprocess_logic(X: list, y: np.ndarray) -> RAW_DATASET_TYPE:
        #     return np.array(X), y

        # with pytest.raises(ValueError):
        #     DatasetBuilder._validate_raw_preprocess_logic(
        #         invalid_arg_type_raw_preprocess_logic,
        #         logger=mock_logger,  # type: ignore
        #     )


class TestValidationTransformLogic:
    def test__validate_transform_logic_valid(self, mocker: MockFixture) -> None:
        def valid_transform_logic(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray],
            y_val: Optional[np.ndarray],
            X_test: np.ndarray,
            y_test: np.ndarray,
        ) -> PROCESSCED_DATASET_TYPE:
            return X_train, y_train, X_val, y_val, X_test, y_test

        mock_logger = mocker.MagicMock()
        DatasetBuilder._validate_transform_logic(
            valid_transform_logic,
            logger=mock_logger,  # type: ignore
        )

        def valid_transform_logic_no_val_data(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
        ) -> PROCESSCED_DATASET_TYPE:
            return X_train, y_train, None, None, X_test, y_test

        mock_logger = mocker.MagicMock()
        DatasetBuilder._validate_transform_logic(
            valid_transform_logic,
            logger=mock_logger,  # type: ignore
        )

    def test__validate_transform_logic_valid_no_typehint(self, mocker: MockFixture) -> None:
        def valid_transform_logic_no_typehint(  # type: ignore
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ):
            return X_train, y_train, X_val, y_val, X_test, y_test

        mock_logger = mocker.MagicMock()
        DatasetBuilder._validate_transform_logic(
            valid_transform_logic_no_typehint,
            logger=mock_logger,  # type: ignore
        )
        mock_logger.warning.assert_called_with(
            "All arguments of the custom raw preprocess function assigned to the type hint."
            "Input and return type validation will be skipped."
        )

    def test__validate_transform_logic_invalid(self, mocker: MockFixture) -> None:
        def invalid_args_transform_logic(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
        ) -> PROCESSCED_DATASET_TYPE:
            return X_train, y_train, X_val, y_val, X_test, y_train

        mock_logger = mocker.MagicMock()
        with pytest.raises(ValueError):
            DatasetBuilder._validate_transform_logic(
                invalid_args_transform_logic,
                logger=mock_logger,  # type: ignore
            )

        def invalid_return_transform_logic(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return X_train, y_train

        with pytest.raises(ValueError):
            DatasetBuilder._validate_transform_logic(
                invalid_return_transform_logic,
                logger=mock_logger,  # type: ignore
            )

        # def invalid_arg_type_transform_logic(
        #     X_train: list,
        #     y_train: np.ndarray,
        #     X_val: np.ndarray,
        #     y_val: np.ndarray,
        #     X_test: np.ndarray,
        #     y_test: np.ndarray,
        # ) -> PROCESSCED_DATASET_TYPE:
        #     return np.array(X_train), y_train, X_val, y_val, X_test, y_test

        # with pytest.raises(ValueError):
        #     DatasetBuilder._validate_transform_logic(
        #         invalid_arg_type_transform_logic,
        #         logger=mock_logger,  # type: ignore
        #     )


GEN_DATA_CONFIG_WITH_VAL = {
    "dataset": {
        "type": "generate",
        "path": {"data": "", "label": ""},
        "random_seed": 42,
        "split": {"train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2, "shuffle": True},
    }
}

GEN_DATA_CONFIG_NO_VAL = {
    "dataset": {
        "type": "generate",
        "path": {"data": "", "label": ""},
        "random_seed": 42,
        "split": {"train_ratio": 0.8, "validation_ratio": 0.0, "test_ratio": 0.2, "shuffle": True},
    }
}


@pytest.fixture(scope="function")
def default_gen_builder(experiment_config: ExperimentConfig) -> DatasetBuilder:
    dataset_config = DatasetConfig(**GEN_DATA_CONFIG_WITH_VAL["dataset"])
    return DatasetBuilder(config=experiment_config.model_copy(update={"dataset": dataset_config}))


@pytest.fixture(scope="function")
def default_gen_builder_no_val(experiment_config: ExperimentConfig) -> DatasetBuilder:
    dataset_config = DatasetConfig(**GEN_DATA_CONFIG_NO_VAL["dataset"])
    return DatasetBuilder(config=experiment_config.model_copy(update={"dataset": dataset_config}))


CUSTOM_CONFIG = {
    "dataset": {
        "type": "generate",
        "path": {"data": "", "label": ""},
        "random_seed": 42,
        "split": {"train_ratio": 0.8, "validation_ratio": 0.0, "test_ratio": 0.2, "shuffle": True},
        "features": None,
        "raw_preprocess_logic": {"module_name": __name__, "implement_name": "custom_raw_preprocess", "params": {}},
        "transform_logic": {"module_name": __name__, "implement_name": "custom_transform", "params": {}},
    }
}


def custom_raw_preprocess(X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
    return X[:50], y[:50]  # extract first 50 samples


def custom_transform(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> PROCESSCED_DATASET_TYPE:
    # all lable change to 1
    y_train = np.ones_like(y_train)
    if y_val is not None:
        y_val = np.ones_like(y_val)
    y_test = np.ones_like(y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture(scope="function")
def custom_builder(experiment_config: ExperimentConfig) -> DatasetBuilder:
    dataset_config = DatasetConfig(**CUSTOM_CONFIG["dataset"])
    return DatasetBuilder(config=experiment_config.model_copy(update={"dataset": dataset_config}))


class TestBuilder:
    def test_load_gen_data(self, default_gen_builder: DatasetBuilder) -> None:
        X, y = default_gen_builder.load()
        assert len(X) == len(y)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_load_file_data(self) -> None:
        # [TODO]: Implement file data loading test
        pass

    def test_split(self, default_gen_builder: DatasetBuilder, default_gen_builder_no_val: DatasetBuilder) -> None:
        X = np.random.rand(100, 2)
        y = np.random.randint(2, size=100)

        # split with validation set
        X_train, y_train, X_val, y_val, X_test, y_test = default_gen_builder.split(X, y)
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)  # type: ignore
        assert len(X_test) == len(y_test)
        assert len(X_train) == 60
        assert len(X_val) == 20  # type: ignore
        assert len(X_test) == 20

        # split without validation set
        X_train, y_train, X_val, y_val, X_test, y_test = default_gen_builder_no_val.split(X, y)
        assert len(X_train) == len(y_train)
        assert X_val is None
        assert y_val is None
        assert len(X_test) == len(y_test)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_raw_preprocess(self, default_gen_builder: DatasetBuilder, custom_builder: DatasetBuilder) -> None:
        X = np.random.rand(100, 2)
        y = np.random.randint(2, size=100)

        # dafault raw preprocess logic is not change the data
        X_p, y_p = default_gen_builder.raw_preprocess(X, y)
        assert (X == X_p).all()
        assert (y == y_p).all()

        # custom raw preprocess logic is extract first 50 samples
        X_p, y_p = custom_builder.raw_preprocess(X, y)
        assert len(X_p) == 50
        assert len(y_p) == 50

    def test_transform(self, default_gen_builder: DatasetBuilder, custom_builder: DatasetBuilder) -> None:
        X_train = np.random.rand(60, 2)
        y_train = np.random.randint(2, size=60)
        X_val = np.random.rand(20, 2)
        y_val = np.random.randint(2, size=20)
        X_test = np.random.rand(20, 2)
        y_test = np.random.randint(2, size=20)

        # dafault transform logic is not change the data
        X_train_t, y_train_t, X_val_t, y_val_y, X_test_t, y_test_t = default_gen_builder.transform(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        assert (X_train == X_train_t).all()
        assert (y_train == y_train_t).all()
        assert (X_val == X_val_t).all()
        assert (y_val == y_val_y).all()
        assert (X_test == X_test_t).all()
        assert (y_test == y_test_t).all()

        # custom transform logic is change all label to 1
        X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t = custom_builder.transform(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        assert (X_train == X_train_t).all()
        assert (y_train_t == np.ones_like(y_train)).all()
        assert (X_val == X_val_t).all()
        assert (y_val_t == np.ones_like(y_val)).all()
        assert (X_test == X_test_t).all()
        assert (y_test_t == np.ones_like(y_test)).all()

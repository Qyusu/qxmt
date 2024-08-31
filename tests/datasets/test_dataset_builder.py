import numpy as np
import pytest

from qxmt.datasets import DatasetBuilder

RAW_DATA_TYPE = np.ndarray
RAW_LABEL_TYPE = np.ndarray
RAW_DATASET_TYPE = tuple[RAW_DATA_TYPE, RAW_LABEL_TYPE]
PROCESSCED_DATASET_TYPE = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class TestValidationPreprocessLogic:
    def test__validate_raw_preprocess_logic_valid(self) -> None:
        def valid_raw_preprocess_logic(X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
            return X, y

        DatasetBuilder._validate_raw_preprocess_logic(valid_raw_preprocess_logic)

    def test__validate_raw_preprocess_logic_invalid(self) -> None:
        def invalid_args_raw_preprocess_logic(X: np.ndarray) -> RAW_DATASET_TYPE:
            return X, y

        with pytest.raises(ValueError):
            DatasetBuilder._validate_raw_preprocess_logic(invalid_args_raw_preprocess_logic)

        def invalid_return_raw_preprocess_logic(
            X: np.ndarray, y: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return X, y, y

        with pytest.raises(ValueError):
            DatasetBuilder._validate_raw_preprocess_logic(invalid_return_raw_preprocess_logic)

        # def invalid_arg_type_raw_preprocess_logic(X: list, y: np.ndarray) -> RAW_DATASET_TYPE:
        #     return np.array(X), y

        # with pytest.raises(ValueError):
        #     DatasetBuilder._validate_raw_preprocess_logic(invalid_arg_type_raw_preprocess_logic)


class TestValidationTransformLogic:
    def test__validate_transform_logic_valid(self) -> None:
        def valid_transform_logic(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
        ) -> PROCESSCED_DATASET_TYPE:
            return X_train, y_train, X_test, y_test

        DatasetBuilder._validate_transform_logic(valid_transform_logic)

    def test__validate_transform_logic_invalid(self) -> None:
        def invalid_args_transform_logic(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
        ) -> PROCESSCED_DATASET_TYPE:
            return X_train, y_train, X_test, y_train

        with pytest.raises(ValueError):
            DatasetBuilder._validate_transform_logic(invalid_args_transform_logic)

        def invalid_return_transform_logic(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return X_train, y_train

        with pytest.raises(ValueError):
            DatasetBuilder._validate_transform_logic(invalid_return_transform_logic)

        # def invalid_arg_type_transform_logic(
        #     X_train: list,
        #     y_train: np.ndarray,
        #     X_test: np.ndarray,
        #     y_test: np.ndarray,
        # ) -> PROCESSCED_DATASET_TYPE:
        #     return np.array(X_train), y_train, X_test, y_test
        #
        # with pytest.raises(ValueError):
        #     DatasetBuilder._validate_transform_logic(invalid_arg_type_transform_logic)


GEN_DATA_CONFIG = {
    "dataset": {
        "type": "generate",
        "path": {"data": "", "label": ""},
        "random_seed": 42,
        "test_size": 0.2,
        "features": None,
    }
}


@pytest.fixture(scope="function")
def default_gen_builder() -> DatasetBuilder:
    return DatasetBuilder(raw_config=GEN_DATA_CONFIG)


CUSTOM_CONFIG = {
    "dataset": {
        "type": "generate",
        "path": {"data": "", "label": ""},
        "random_seed": 42,
        "test_size": 0.2,
        "features": None,
        "raw_preprocess_logic": {"module_name": __name__, "function_name": "custom_raw_preprocess", "params": {}},
        "transform_logic": {"module_name": __name__, "function_name": "custom_transform", "params": {}},
    }
}


def custom_raw_preprocess(X: np.ndarray, y: np.ndarray) -> RAW_DATASET_TYPE:
    # extract first 50 samples
    return X[:50], y[:50]


def custom_transform(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> PROCESSCED_DATASET_TYPE:
    # all lable change to 1
    y_train = np.ones_like(y_train)
    y_test = np.ones_like(y_test)
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="function")
def custom_builder() -> DatasetBuilder:
    return DatasetBuilder(raw_config=CUSTOM_CONFIG)


class TestBuilder:
    def test_load_gen_data(self, default_gen_builder: DatasetBuilder) -> None:
        X, y = default_gen_builder.load()
        assert len(X) == len(y)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_load_file_data(self) -> None:
        # [TODO]: Implement file data loading test
        pass

    def test_split(self, default_gen_builder: DatasetBuilder) -> None:
        X = np.random.rand(100, 2)
        y = np.random.randint(2, size=100)

        X_train, y_train, X_test, y_test = default_gen_builder.split(X, y)
        assert len(X_train) == len(y_train)
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
        X_train = np.random.rand(80, 2)
        y_train = np.random.randint(2, size=80)
        X_test = np.random.rand(20, 2)
        y_test = np.random.randint(2, size=20)

        # dafault transform logic is not change the data
        X_train_t, y_train_t, X_test_t, y_test_t = default_gen_builder.transform(X_train, y_train, X_test, y_test)
        assert (X_train == X_train_t).all()
        assert (y_train == y_train_t).all()
        assert (X_test == X_test_t).all()
        assert (y_test == y_test_t).all()

        # custom transform logic is change all label to 1
        X_train_t, y_train_t, X_test_t, y_test_t = custom_builder.transform(X_train, y_train, X_test, y_test)
        assert (X_train == X_train_t).all()
        assert (y_train_t == np.ones_like(y_train)).all()
        assert (X_test == X_test_t).all()
        assert (y_test_t == np.ones_like(y_test)).all()

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockFixture

from qxmt.datasets.tfds.loader import TFDSDataLoader, _import_tfds


class TestTFDSDataLoader:
    def test_import_tfds_error_message(self, mocker: MockFixture) -> None:
        mocker.patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'tensorflow_datasets'"),
        )

        with pytest.raises(ImportError, match=r"pip install 'qxmt\[tfds\]'"):
            _import_tfds()

    def test_init(self) -> None:
        loader = TFDSDataLoader(name="mnist", split="test", save_path="data/mnist.npz")
        assert loader.name == "mnist"
        assert loader.split == "test"
        assert loader.save_path == "data/mnist.npz"
        assert loader.return_format == "numpy"
        assert loader.download is True
        assert loader.shuffle_files is False
        assert loader.flatten is True

    def test_load(self, mocker: MockFixture) -> None:
        mock_tfds = mocker.Mock()
        mock_dataset = mocker.Mock()
        mock_tfds.load.return_value = mock_dataset
        mock_tfds.as_numpy.return_value = [
            (np.array([1, 2]), np.array(0)),
            (np.array([3, 4]), np.array(1)),
        ]
        mocker.patch("importlib.import_module", return_value=mock_tfds)

        loader = TFDSDataLoader(name="mnist", split="train[:2]", download=False, shuffle_files=True)
        X, y = loader.load()

        mock_tfds.load.assert_called_once_with(
            "mnist",
            split="train[:2]",
            as_supervised=True,
            download=False,
            shuffle_files=True,
        )
        mock_tfds.as_numpy.assert_called_once_with(mock_dataset)
        assert np.allclose(X, np.array([[1, 2], [3, 4]]))
        assert np.allclose(y, np.array([0, 1]))

    def test_load_with_flatten(self, mocker: MockFixture) -> None:
        mock_tfds = mocker.Mock()
        mock_dataset = mocker.Mock()
        mock_tfds.load.return_value = mock_dataset
        mock_tfds.as_numpy.return_value = [
            (np.ones((2, 2, 1)), np.array(0)),
            (np.zeros((2, 2, 1)), np.array(1)),
        ]
        mocker.patch("importlib.import_module", return_value=mock_tfds)

        loader = TFDSDataLoader(name="mnist", flatten=True)
        X, y = loader.load()

        assert X.shape == (2, 4)
        assert np.allclose(y, np.array([0, 1]))

    def test_flatten_features_keeps_2d_data(self) -> None:
        loader = TFDSDataLoader(name="iris", flatten=True)
        X = np.ones((3, 4))

        assert loader._flatten_features(X).shape == (3, 4)

    def test_load_multiple_splits(self, mocker: MockFixture) -> None:
        mock_tfds = mocker.Mock()
        mock_train_dataset = mocker.Mock()
        mock_test_dataset = mocker.Mock()
        mock_tfds.load.return_value = [mock_train_dataset, mock_test_dataset]
        mock_tfds.as_numpy.side_effect = [
            [(np.array([1, 2]), np.array(0))],
            [(np.array([3, 4]), np.array(1))],
        ]
        mocker.patch("importlib.import_module", return_value=mock_tfds)

        loader = TFDSDataLoader(name="mnist", split=["train", "test"])
        X, y = loader.load()

        mock_tfds.load.assert_called_once_with(
            "mnist",
            split=["train", "test"],
            as_supervised=True,
            download=True,
            shuffle_files=False,
        )
        assert mock_tfds.as_numpy.call_count == 2
        assert np.allclose(X, np.array([[1, 2], [3, 4]]))
        assert np.allclose(y, np.array([0, 1]))

    def test_load_with_data_dir(self, mocker: MockFixture, tmp_path: Path) -> None:
        mock_tfds = mocker.Mock()
        mock_tfds.load.return_value = mocker.Mock()
        mock_tfds.as_numpy.return_value = [(np.array([1]), np.array(0))]
        mocker.patch("importlib.import_module", return_value=mock_tfds)

        loader = TFDSDataLoader(name="mnist", data_dir=tmp_path)
        loader.load()

        assert mock_tfds.load.call_args.kwargs["data_dir"] == str(tmp_path)

    def test_invalid_return_format(self) -> None:
        loader = TFDSDataLoader(name="mnist", return_format="pandas")

        with pytest.raises(ValueError, match="Unsupported return format"):
            loader.load()

    def test_save_dataset(self, tmp_path: Path) -> None:
        X = np.random.rand(3, 2)
        y = np.array([0, 1, 0])

        loader = TFDSDataLoader(name="mnist", save_path=tmp_path / "dataset.npz")
        loader._save_dataset((X, y))
        loaded_data = np.load(tmp_path / "dataset.npz")
        assert np.allclose(loaded_data["X"], X)
        assert np.allclose(loaded_data["y"], y)

        loader = TFDSDataLoader(name="mnist", save_path=tmp_path / "dataset.npy")
        loader._save_dataset((X, y))
        assert np.allclose(np.load(tmp_path / "dataset_X.npy"), X)
        assert np.allclose(np.load(tmp_path / "dataset_y.npy"), y)

        loader = TFDSDataLoader(name="mnist", save_path=None)
        with pytest.raises(ValueError, match="Save path is not specified"):
            loader._save_dataset((X, y))

        loader = TFDSDataLoader(name="mnist", save_path=tmp_path / "dataset.csv")
        with pytest.raises(ValueError, match="Unsupported save format"):
            loader._save_dataset((X, y))

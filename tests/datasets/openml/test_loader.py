from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockFixture

from qxmt.datasets.openml.loader import OpenMLDataLoader


class TestOpenMLDataLoader:
    def test_init(self, mocker: MockFixture) -> None:
        mocker.patch("qxmt.datasets.openml.loader.OpenMLDataLoader._get_dataset_id", return_value=554)
        # set id
        loader = OpenMLDataLoader(id=554)
        assert loader.name is None
        assert loader.id == 554
        assert loader.save_path is None
        assert loader.return_format == "numpy"
        assert loader.use_cache is True

        # set only name
        loader = OpenMLDataLoader(name="mnist_784")
        assert loader.name == "mnist_784"
        assert loader.id == 554
        assert loader.save_path is None
        assert loader.return_format == "numpy"
        assert loader.use_cache is True

        # set name and id, this case prioritizes id over name
        loader = OpenMLDataLoader(name="mnist_784", id=555)
        assert loader.name == "mnist_784"
        assert loader.id == 555
        assert loader.save_path is None
        assert loader.return_format == "numpy"
        assert loader.use_cache is True

        # erro case, both name and id are None
        with pytest.raises(ValueError):
            OpenMLDataLoader(name=None, id=None)

    def test_get_dataset_id(self, mocker: MockFixture) -> None:
        mocker.patch(
            "openml.datasets.list_datasets",
            return_value=pd.DataFrame(
                {"name": ["mnist_784", "mnist_784", "Fashion-MNIST"], "version": [1, 2, 1], "did": [553, 554, 40996]}
            ),
        )
        loader = OpenMLDataLoader(name="mnist_784")
        assert loader._get_dataset_id() == 554

        # error case, dataset not found
        with pytest.raises(ValueError):
            loader = OpenMLDataLoader(name="not_exist")
            loader._get_dataset_id()

    def test_load(self, mocker: MockFixture, tmp_path: Path) -> None:
        mock_dataset_instance = mocker.Mock()
        mock_dataset_instance.get_data.return_value = (
            pd.DataFrame(np.random.rand(3, 3), columns=["A", "B", "C"]),
            pd.Series([1, 0, 1], name="target"),
            None,
            ["A", "B", "C", "target"],
        )
        mocker.patch("openml.datasets.get_dataset", return_value=mock_dataset_instance)

        loader = OpenMLDataLoader(id=554, return_format="pandas", save_path=tmp_path / "dataset.csv")
        result = loader.load()
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 4
        assert (tmp_path / "dataset.csv").exists()

    def test_format_return_data(self) -> None:
        X = pd.DataFrame(np.random.rand(10, 3), columns=["feature_" + str(i) for i in range(3)])
        y = pd.DataFrame({"label": np.random.randint(0, 2, 10)})

        # case: return_format="numpy", y is not None
        loader = OpenMLDataLoader(id=554, return_format="numpy")
        result = loader._format_return_data(X, y)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

        # case: return_format="numpy", y is None
        loader = OpenMLDataLoader(id=554, return_format="numpy")
        result = loader._format_return_data(X, None)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert result[1] is None

        # case: return_format="pandas", y is not None
        loader = OpenMLDataLoader(id=554, return_format="pandas")
        result = loader._format_return_data(X, y)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 4

        # case: return_format="pandas", y is None
        loader = OpenMLDataLoader(id=554, return_format="pandas")
        result = loader._format_return_data(X, None)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3

        # case: return_format is invalid, raise ValueError
        loader = OpenMLDataLoader(id=554, return_format="invalid")
        with pytest.raises(ValueError):
            loader._format_return_data(X, y)

    def test_save_dataset(self, tmp_path: Path) -> None:
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10)
        data_X_array = (X, None)
        data_Xy_array = (X, y)

        # case: return_format="numpy", extension=".npz", y is not None
        loader = OpenMLDataLoader(id=554, save_path=tmp_path / "dataset.npz", return_format="numpy")
        loader._save_dataset(data_Xy_array)
        assert (tmp_path / "dataset.npz").exists()

        loaded_data = np.load(tmp_path / "dataset.npz")
        assert np.allclose(loaded_data["X"], X)
        assert np.allclose(loaded_data["y"], y)

        # case: return_format="numpy", extension=".npz", y is None
        loader = OpenMLDataLoader(id=554, save_path=tmp_path / "dataset.npz", return_format="numpy")
        loader._save_dataset(data_X_array)
        assert (tmp_path / "dataset.npz").exists()

        loaded_data = np.load(tmp_path / "dataset.npz")
        assert np.allclose(loaded_data["X"], X)
        assert "y" not in loaded_data.files

        # case: return_format="numpy", extension=".npy", y is not None
        loader = OpenMLDataLoader(id=554, save_path=tmp_path / "dataset.npy", return_format="numpy")
        loader._save_dataset(data_Xy_array)
        assert not (tmp_path / "dataset.npy").exists()

        loaded_X_data = np.load(tmp_path / "dataset_X.npy")
        loaded_y_data = np.load(tmp_path / "dataset_y.npy")
        assert np.allclose(loaded_X_data, X)
        assert np.allclose(loaded_y_data, y)

        # case: return_format="numpy", extension=".npy", y is None
        loader = OpenMLDataLoader(id=554, save_path=tmp_path / "dataset.npy", return_format="numpy")
        loader._save_dataset(data_X_array)
        assert (tmp_path / "dataset.npy").exists()

        loaded_data = np.load(tmp_path / "dataset.npy")
        assert np.allclose(loaded_data, X)

        data_df = pd.DataFrame(np.random.rand(10, 3), columns=["feature_" + str(i) for i in range(3)])

        # case: return_format="pandas", extension=".csv"
        loader = OpenMLDataLoader(id=554, save_path=tmp_path / "dataset.csv", return_format="pandas")
        loader._save_dataset(data_df)
        assert (tmp_path / "dataset.csv").exists()

        loaded_data = pd.read_csv(tmp_path / "dataset.csv")
        assert loaded_data.shape[1] == 3

        # case: return_format="pandas", extension=".tsv"
        loader = OpenMLDataLoader(id=554, save_path=tmp_path / "dataset.tsv", return_format="pandas")
        loader._save_dataset(data_df)
        assert (tmp_path / "dataset.tsv").exists()

        loaded_data = pd.read_csv(tmp_path / "dataset.tsv", sep="\t")
        assert loaded_data.shape[1] == 3

        # case: save_path is None, raise ValueError
        loader = OpenMLDataLoader(id=554, save_path=None)
        with pytest.raises(ValueError):
            loader._save_dataset(data_df)

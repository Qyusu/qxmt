from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qxmt.datasets.file.loader import FileDataLoader


class TestFileDataLoader:
    def test__init__(self) -> None:
        laoder = FileDataLoader(data_path="data.npy", label_path="label.npy", label_name="target")
        assert isinstance(laoder.data_path, Path)
        assert isinstance(laoder.label_path, Path)
        assert laoder.label_name == "target"

        laoder = FileDataLoader(data_path="data.npy")
        assert isinstance(laoder.data_path, Path)
        assert laoder.label_path is None
        assert laoder.label_name is None

    def test_load_npy(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.npy"
        label_path = tmp_path / "label.npy"
        data = [1, 2, 3]
        label = [0, 1, 0]
        np.save(data_path, data)
        np.save(label_path, label)

        loader = FileDataLoader(data_path=data_path, label_path=label_path)
        X, y = loader.load()
        assert np.allclose(X, data)
        assert np.allclose(y, label)

    def test_load_npz(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.npz"
        data = [1, 2, 3]
        label = [0, 1, 0]
        np.savez(data_path, X=data, y=label)

        loader = FileDataLoader(data_path=data_path)
        X, y = loader.load()
        assert np.allclose(X, data)
        assert np.allclose(y, label)

        # case: key is not matched
        not_match_data_path = tmp_path / "not_match_data.npz"
        np.savez(not_match_data_path, X=data, label=label)
        loader = FileDataLoader(data_path=not_match_data_path)
        with pytest.raises(ValueError):
            loader.load()

    def test_load_two_file_csv(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.csv"
        label_path = tmp_path / "label.csv"
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [0, 1, 0]})
        label = pd.DataFrame({"target": [0, 1, 0]})
        data.to_csv(data_path, index=False)
        label.to_csv(label_path, index=False)

        # valid case: label_name is None
        loader = FileDataLoader(data_path=data_path, label_path=label_path)
        X, y = loader.load()
        assert np.allclose(X, data.values)
        assert np.allclose(y, label.values)

        # invalid case: label_name is not None
        loader = FileDataLoader(data_path=data_path, label_path=label_path, label_name="target")
        with pytest.raises(ValueError):
            X, y = loader.load()

    def test_load_one_file_csv(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.csv"
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [0, 1, 0], "target": [0, 1, 0]})
        data.to_csv(data_path, index=False)

        # valid case: label_name is not None
        loader = FileDataLoader(data_path=data_path, label_path=None, label_name="target")
        X, y = loader.load()
        assert np.allclose(X, data.drop(columns=["target"]).values)
        assert np.allclose(y, np.array(data["target"].values))

        # invalid case: label_name is None
        loader = FileDataLoader(data_path=data_path, label_path=None, label_name=None)
        with pytest.raises(ValueError):
            X, y = loader.load()

    def test_load_two_file_tsv(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.tsv"
        label_path = tmp_path / "label.tsv"
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [0, 1, 0]})
        label = pd.DataFrame({"target": [0, 1, 0]})
        data.to_csv(data_path, sep="\t", index=False)
        label.to_csv(label_path, sep="\t", index=False)

        # valid case: label_name is None
        loader = FileDataLoader(data_path=data_path, label_path=label_path)
        X, y = loader.load()
        assert np.allclose(X, data.values)
        assert np.allclose(y, label.values)

        # invalid case: label_name is not None
        loader = FileDataLoader(data_path=data_path, label_path=label_path, label_name="target")
        with pytest.raises(ValueError):
            X, y = loader.load()

    def test_load_one_file_tsv(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.tsv"
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [0, 1, 0], "target": [0, 1, 0]})
        data.to_csv(data_path, sep="\t", index=False)

        # valid case: label_name is not None
        loader = FileDataLoader(data_path=data_path, label_path=None, label_name="target")
        X, y = loader.load()
        assert np.allclose(X, data.drop(columns=["target"]).values)
        assert np.allclose(y, np.array(data["target"].values))

        # invalid case: label_name is None
        loader = FileDataLoader(data_path=data_path, label_path=None, label_name=None)
        with pytest.raises(ValueError):
            X, y = loader.load()

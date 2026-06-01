import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np


def _import_tfds() -> Any:
    try:
        return importlib.import_module("tensorflow_datasets")
    except ImportError as e:
        raise ImportError(
            "TensorFlow Datasets require the optional dependency 'tensorflow-datasets'. "
            "Install it with: pip install 'qxmt[tfds]'"
        ) from e


class TFDSDataLoader:
    """
    This class loads a supervised dataset from TensorFlow Datasets (https://github.com/tensorflow/datasets) and converts it to numpy arrays.

    Supported return formats:
    - numpy: return as a tuple of numpy arrays.

    Supported save formats:
    - numpy: .npz, .npy
    """

    def __init__(
        self,
        name: str,
        split: str | Sequence[str] = "train",
        save_path: Optional[str | Path] = None,
        return_format: str = "numpy",
        data_dir: Optional[str | Path] = None,
        download: bool = True,
        shuffle_files: bool = False,
    ) -> None:
        self.name = name
        self.split = split
        self.save_path = save_path
        self.return_format = return_format.lower()
        self.data_dir = data_dir
        self.download = download
        self.shuffle_files = shuffle_files

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load a supervised TensorFlow Dataset and return features and labels as numpy arrays."""
        if self.return_format not in ["numpy", "array"]:
            raise ValueError(f"Unsupported return format: {self.return_format}")

        tfds = _import_tfds()
        load_kwargs: dict[str, Any] = {
            "split": self.split,
            "as_supervised": True,
            "download": self.download,
            "shuffle_files": self.shuffle_files,
        }
        if self.data_dir is not None:
            load_kwargs["data_dir"] = str(self.data_dir)

        datasets = tfds.load(self.name, **load_kwargs)
        if isinstance(datasets, list):
            data = []
            for dataset in datasets:
                data.extend(tfds.as_numpy(dataset))
        else:
            data = list(tfds.as_numpy(datasets))
        X = np.asarray([example[0] for example in data])
        y = np.asarray([example[1] for example in data])

        if self.save_path:
            self._save_dataset((X, y))

        return X, y

    def _save_dataset(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        if self.save_path is None:
            raise ValueError("Save path is not specified.")

        save_path = Path(self.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        extention = save_path.suffix
        if self.return_format in ["numpy", "array"] and extention == ".npz":
            np.savez(save_path, X=cast(np.ndarray, data[0]), y=cast(np.ndarray, data[1]))
        elif self.return_format in ["numpy", "array"] and extention == ".npy":
            np.save(
                save_path.with_name(save_path.stem + "_X" + save_path.suffix),
                cast(np.ndarray, data[0]),
            )
            np.save(
                save_path.with_name(save_path.stem + "_y" + save_path.suffix),
                cast(np.ndarray, data[1]),
            )
        else:
            raise ValueError(f'Unsupported save format: data_format="{self.return_format}", extention="{extention}"')

from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd


class FileDataLoader:
    """
    This class loads the data and label from the multi format file.
    The loaded data and label are returned as numpy arrays (X and y).

    Input data schema supports two patterns:
    1. data and label are defined in separate files.
    2. data and label are defined in a single file. In this case, the label name must be defined.

    Supported file formats:
    - .npy, .npz, .csv, .tsv

    Examples:
        >>> loader = FileDataLoader(data_path="data.npy", label_path="label.npy")
        >>> X, y = loader.load()
        >>> loader = FileDataLoader(data_path="data.npz")
        >>> X, y = loader.load()
        >>> loader = FileDataLoader(data_path="data.csv", label_path="label.csv")
        >>> X, y = loader.load()
        >>> loader = FileDataLoader(data_path="data.csv", label_name="target")
        >>> X, y = loader.load()
        >>> loader = FileDataLoader(data_path="data.tsv", label_path="label.tsv")
        >>> X, y = loader.load()
        >>> loader = FileDataLoader(data_path="data.tsv", label_name="target")
        >>> X, y = loader.load()
    """

    def __init__(
        self,
        data_path: str | Path,
        label_path: Optional[str | Path] = None,
        label_name: Optional[str] = None,
    ) -> None:
        """Initialize the FileDataLoader.

        Args:
            data_path (str | Path): path to the data file.
            label_path (Optional[str  |  Path], optional): path to the label file. Defaults to None.
            label_name (Optional[str], optional): label name in the dataset. Defaults to None.
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path) if label_path is not None else None
        self.label_name = label_name

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the data and label from the file path.
        The file format is determined by the extension of the file path.

        Supported file formats:
        - numpy: .npy, .npz
        - pandas: .csv, .tsv

        Two input patterns exist:
        1. "data_path" and "label_path" are defined.
        "label_name" is not needed, because the label is loaded from the file.
        2. "data_path" and "label_name" are defined.
        "label_path" is not needed, because the label data include in the data file.

        Returns:
            tuple[np.ndarray, np.ndarray]: loaded data and label as numpy arrays.

        Raises:
            ValueError: data and label file extensions do not match
            ValueError: Data or label key is not matched in the npz file.
            ValueError: Label defined in data of "label_path", not need to define "label_name".
            ValueError: "Data and label are expected to be contained in the single file defined in "data_path"
            ValueError: Label name is not found in the dataset.
            ValueError: unsupported file extension
        """
        data_path = Path(self.data_path)
        data_extension = data_path.suffix

        # label path and extension check only executed when label is not None
        if self.label_path is not None:
            label_path = Path(self.label_path)
            label_extension = label_path.suffix
            if data_extension != label_extension:
                raise ValueError("Data and label file extensions do not match.")

        match data_extension:
            case ".npy":
                # input two npy file paths from config
                X = np.load(data_path, allow_pickle=True)
                y = np.load(label_path, allow_pickle=True)
            case ".npz" if self.label_path is None:
                # input one npz file path from config
                data = np.load(data_path, allow_pickle=True)
                X = data.get("X", None)
                y = data.get("y", None)
                if X is None or y is None:
                    raise ValueError(
                        """Data or label key is not matched in the npz file.
                        Data key must be set as 'X' and label key must be set as 'y'."""
                    )
            case ".csv" if self.label_path is not None:
                # input two csv file paths from config
                if self.label_name is not None:
                    raise ValueError('Label defined in data of "label_path", not need to define "label_name".')
                X = pd.read_csv(data_path, sep=",").values
                y = pd.read_csv(label_path, sep=",").values
            case ".csv" if self.label_path is None:
                # input one csv file path from config
                if self.label_path is not None:
                    raise ValueError(
                        """Data and label are expected to be contained in the single file defined in "data_path"
                        , but "label_path" is also defined."""
                    )
                data = pd.read_csv(data_path, sep=",")
                if self.label_name not in data.columns:
                    raise ValueError(f'Label name "{self.label_name}" is not found in the dataset.')
                X = data.drop(columns=[self.label_name]).values
                y = cast(np.ndarray, data[self.label_name].values)
            case ".tsv" if self.label_path is not None:
                # input two tsv file paths from config
                if self.label_name is not None:
                    raise ValueError('Label defined in data of "label_path", not need to define "label_name".')
                X = pd.read_csv(data_path, sep="\t").values
                y = pd.read_csv(label_path, sep="\t").values
            case ".tsv" if self.label_path is None:
                # input one tsv file path from config
                if self.label_path is not None:
                    raise ValueError(
                        """Data and label are expected to be contained in the single file defined in "data_path"
                        , but "label_path" is also defined."""
                    )
                data = pd.read_csv(data_path, sep="\t")
                if self.label_name not in data.columns:
                    raise ValueError(f'Label name "{self.label_name}" is not found in the dataset.')
                X = data.drop(columns=[self.label_name]).values
                y = data[self.label_name].values
            case _:
                raise ValueError(f'Unsupported file extension: "{data_extension}"')

        return X, np.array(y)

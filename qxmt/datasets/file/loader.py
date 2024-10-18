from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd


class FileDataLoader:
    def __init__(self, data_path: str | Path, label_path: Optional[str | Path], label_name: Optional[str]) -> None:
        self.data_path = Path(data_path)
        self.label_path = Path(label_path) if label_path is not None else None
        self.label_name = label_name if label_name is not None else "label"

    def load(self) -> tuple[np.ndarray, np.ndarray]:
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
            case ".csv" if self.label_path is None:
                # input two csv file paths from config
                X = pd.read_csv(data_path, sep=",").values
                y = pd.read_csv(label_path, sep=",").values
            case ".csv" if self.label_path is not None:
                # input one csv file path from config
                data = pd.read_csv(data_path, sep=",")
                if self.label_name not in data.columns:
                    raise ValueError(f'Label name "{self.label_name}" is not found in the dataset.')
                X = data.drop(columns=[self.label_name]).values
                y = cast(np.ndarray, data[self.label_name].values)
            case ".tsv" if self.label_path is None:
                # input two tsv file paths from config
                X = pd.read_csv(data_path, sep="\t").values
                y = pd.read_csv(label_path, sep="\t").values
            case ".tsv" if self.label_path is not None:
                # input one tsv file path from config
                data = pd.read_csv(data_path, sep="\t")
                if self.label_name not in data.columns:
                    raise ValueError(f'Label name "{self.label_name}" is not found in the dataset.')
                X = data.drop(columns=[self.label_name]).values
                y = data[self.label_name].values
            case _:
                raise ValueError(f'Unsupported file extension: "{data_extension}"')

        return cast(np.ndarray, X), cast(np.ndarray, y)

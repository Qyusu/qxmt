from logging import Logger
from pathlib import Path
from typing import Optional, cast

import numpy as np
import openml
import pandas as pd

from qxmt.decorators import retry_on_exception
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class OpenMLDataLoader:
    """
    This class loads a dataset from OpenML (https://www.openml.org/) and converts it to the specified return format.
    OpenML is an online platform for sharing datasets and machine learning experiments.
    Identifier of the dataset can be specified by the dataset name or ID.
    If the name is specified, search the OpenML database to get the dataset ID and load the latest version.
    The dataset ID can be search in officail OpenML website (https://www.openml.org/search).
    ID has a higher priority than the name.

    Supported return formats:
    - numpy: return as a tuple of numpy arrays.
    - pandas: return as a pandas DataFrame.

    Supported save formats:
    - numpy: .npz, .npy
    - pandas: .csv, .tsv

    Examples:

        Load the dataset "mnist_784" from OpenML and save it as a numpy file.

        ::

            >>> loader = OpenMLDataLoader(name="mnist_784", save_path="data/mnist.npz")
            >>> X, y = loader.load()
            X = np.array([[0, 0, 0, ..., 0, 0, 0], ..]), y = np.array([5, 0, 4, ..., 4, 5, 6])

        Load the dataset id=554 ("mnist_784") from OpenML and save it as a pandas DataFrame.

        ::

            >>> loader = OpenMLDataLoader(id=554, save_path="data/mnist.csv", return_format="pandas")
            >>> data = loader.load()
            data = pd.DataFrame([[0, 0, 0, ..., 0, 0, 0], .., [5, 0, 4, ..., 4, 5, 6]])
    """

    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        return_format: str = "numpy",
        use_cache: bool = True,
        logger: Logger = LOGGER,
    ) -> None:
        """Initialize the OpenML dataset loader.

        Args:
            dataset_identifier (str | int): dataset name or ID. If the name is specified, the latest version is used.
            save_path (Optional[str  |  Path], optional):
                save path for the loaded dataset.
                If the value is None, the dataset is not saved. Defaults to None.
            return_format (str, optional): return format of the loaded dataset. Defaults to "numpy".
        """
        if (name is None) and (id is None):
            raise ValueError("Either dataset 'name' or 'id' must be specified.")
        self.name = name
        self.id = id if id is not None else self._get_dataset_id()
        self.save_path = save_path
        self.return_format = return_format.lower()
        self.use_cache = use_cache
        self.logger = logger

    @retry_on_exception(retries=3, delay=5)
    def _get_dataset_id(self) -> int:
        """Get the dataset ID from the dataset name by searching the OpenML database.

        Raises:
            ValueError: dataset that matches the name is not found in the OpenML database.

        Returns:
            int: mathced dataset ID
        """
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        matches = datasets[datasets["name"] == self.name]
        if not matches.empty:
            # get the dataset id of the latest version
            matches = matches.sort_values("version", ascending=False)
            return int(matches.iloc[0]["did"])
        else:
            raise ValueError(f"Dataset '{self.name}' not found on OpenML.")

    def load(self) -> tuple[np.ndarray, np.ndarray | None] | pd.DataFrame:
        """Load the dataset from OpenML. Then, convert the dataset to the specified return format.
        The loaded dataset is saved to the specified path if it is not None.

        Raises:
            ValueError: unsupported return format

        Returns:
            tuple[np.ndarray, np.ndarray | None] | pd.DataFrame: loaded dataset
        """
        dataset = openml.datasets.get_dataset(
            self.id,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
            force_refresh_cache=not self.use_cache,
        )
        self.logger.debug(f"Loading dataset: {dataset.name}, ID: {self.id}")

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        data = self._format_return_data(cast(pd.DataFrame, X), cast(pd.DataFrame, y))

        if self.save_path:
            self._save_dataset(data)

        return data

    def _format_return_data(
        self, X: pd.DataFrame, y: pd.DataFrame | None
    ) -> tuple[np.ndarray, np.ndarray | None] | pd.DataFrame:
        """Format the loaded dataset to the specified return format.
        Return format defined in the class attribute `return_format`.
        supported formats:
        - numpy: return as a tuple of numpy arrays.
        - pandas: return as a pandas DataFrame.

        Args:
            X (pd.DataFrame): DataFrame of features.
            y (pd.DataFrame | None): DataFrame of labels.

        Raises:
            ValueError: unsupported return format

        Returns:
            tuple[np.ndarray, np.ndarray | None] | pd.DataFrame: formated dataset for the specified return format.
        """
        if self.return_format in ["numpy", "array"]:
            X_array = X.to_numpy()
            y_array = y.to_numpy() if y is not None else None
            data = (X_array, y_array)
        elif self.return_format in ["pandas", "dataframe"]:
            if y is None:
                data = X
            else:
                data = pd.concat([X, y], axis=1)
        else:
            raise ValueError(f"Unsupported return format: {self.return_format}")

        return data

    def _save_dataset(self, data: tuple[np.ndarray, np.ndarray | None] | pd.DataFrame) -> None:
        """Save the loaded dataset to the specified path.
        Supported save formats:
        - numpy: .npz, .npy
        - pandas: .csv, .tsv

        Args:
            data (tuple[np.ndarray, np.ndarray  |  None] | pd.DataFrame): dataset object to save.

        Raises:
            ValueError: save path is not specified.
            ValueError: not supported save format
        """
        if self.save_path is None:
            raise ValueError("Save path is not specified.")

        save_path = Path(self.save_path)
        # create the directory if it does not exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        extention = save_path.suffix
        if self.return_format in ["numpy", "array"] and extention == ".npz":
            if data[1] is not None:
                np.savez(save_path, X=cast(np.ndarray, data[0]), y=cast(np.ndarray, data[1]))
            else:
                np.savez(save_path, X=cast(np.ndarray, data[0]))
        elif self.return_format in ["numpy", "array"] and extention == ".npy":
            if data[1] is not None:
                np.save(save_path.with_name(save_path.stem + "_X" + save_path.suffix), cast(np.ndarray, data[0]))
                np.save(save_path.with_name(save_path.stem + "_y" + save_path.suffix), cast(np.ndarray, data[1]))
            else:
                np.save(save_path, cast(np.ndarray, data[0]))
        elif self.return_format in ["pandas", "dataframe"] and extention == ".csv":
            pd.DataFrame(data).to_csv(save_path, sep=",", index=False)
        elif self.return_format in ["pandas", "dataframe"] and extention == ".tsv":
            pd.DataFrame(data).to_csv(save_path, sep="\t", index=False)
        else:
            raise ValueError(f'Unsupported save format: data_format="{self.return_format}", extention="{extention}"')

        self.logger.debug(f"Saved dataset to: {save_path.parent}")

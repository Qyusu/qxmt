from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from qxmt.datasets.schema import Dataset
from qxmt.visualization.graph_settings import _create_class_labels, _create_colors

DEFAULT_FEATURE_COLS = ["feature_1", "feature_2"]


def plot_2d_dataset(
    dataset: Dataset,
    colors: Optional[dict[int, str]] = None,
    class_labels: Optional[dict[int, str]] = None,
    save_path: Optional[str | Path] = None,
    **kwargs: dict[str, Any],
) -> None:
    """Ploat dataset on 2D plane.

    Args:
        dataset (Dataset): Dataset object. It contains train and test data.
        colors (Optional[dict[int, str]], optional): color of each class. Defaults to None.
        class_labels (Optional[dict[int, str]], optional): label of each class. Defaults to None.
        save_path (Optional[str], optional): save path of graph. Defaults to None.
        **kwargs (dict[str, Any]): additional arguments for plot.
            train_title (str, optional): title of train dataset. Defaults to "Train Dataset".
            test_title (str, optional): title of test dataset. Defaults to "Test Dataset".
    """
    if dataset.config.features is not None:
        feature_cols = dataset.config.features
    else:
        feature_cols = DEFAULT_FEATURE_COLS

    y_all = np.concatenate([dataset.y_train, dataset.y_test])
    if colors is None:
        colors = _create_colors(y_all)

    if class_labels is None:
        class_labels = _create_class_labels(y_all)

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.subplot(1, 2, 1)
    for class_value in np.unique(dataset.y_train):
        subset = dataset.X_train[np.where(dataset.y_train == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors.get(class_value),
            label=class_labels.get(class_value),
        )
        plt.xlabel(f"{feature_cols[0]}")
        plt.ylabel(f"{feature_cols[1]}")
        plt.legend(title="Class")
        plt.title(str(kwargs.get("train_title", "Train Dataset")))

    plt.subplot(1, 2, 2)
    for class_value in np.unique(dataset.y_test):
        subset = dataset.X_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors.get(class_value),
            label=class_labels.get(class_value),
        )
        plt.xlabel(f"{feature_cols[0]}")
        plt.ylabel(f"{feature_cols[1]}")
        plt.legend(title="Class")
        plt.title(str(kwargs.get("test_title", "Test Dataset")))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

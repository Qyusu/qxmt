from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qxmt.datasets.schema import Dataset

DEFAULT_FEATURE_COLS = ["feature_1", "feature_2"]


def _create_colors(y: np.ndarray) -> dict[int, str]:
    return {class_value: f"C{i}" for i, class_value in enumerate(np.unique(y))}


def _create_class_labels(y: np.ndarray) -> dict[int, str]:
    return {class_value: str(class_value) for class_value in np.unique(y)}


def plot_2d_dataset(
    dataset: Dataset,
    colors: Optional[dict[int, str]] = None,
    class_labels: Optional[dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Ploat dataset on 2D plane.

    Args:
        dataset (Dataset): Dataset object. It contains train and test data.
        colors (Optional[dict[int, str]], optional): color of each class. Defaults to None.
        class_labels (Optional[dict[int, str]], optional): label of each class. Defaults to None.
        save_path (Optional[str], optional): save path of graph. Defaults to None.
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
            c=colors[class_value],
            label=class_labels[class_value],
        )
        plt.xlabel(f"{feature_cols[0]}")
        plt.ylabel(f"{feature_cols[1]}")
        plt.legend()
        plt.title("Train Dataset")

    plt.subplot(1, 2, 2)
    for class_value in np.unique(dataset.y_test):
        subset = dataset.X_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors[class_value],
            label=class_labels[class_value],
        )
        plt.xlabel(f"{feature_cols[0]}")
        plt.ylabel(f"{feature_cols[1]}")
        plt.legend()
        plt.title("Test Dataset")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

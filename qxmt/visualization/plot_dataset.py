from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qxmt.datasets import Dataset

DEFAULT_FEATURE_COLS = ["feature_1", "feature_2"]


def plot_2d_dataset(
    dataset: Dataset,
    colors: Optional[dict[int, tuple]] = None,
    class_labels: Optional[dict[int, str]] = None,
    train_title: str = "Train Dataset",
    test_title: str = "Test Dataset",
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Ploat dataset on 2D plane.

    Args:
        dataset (Dataset): Dataset object. It contains train and test data.
        colors (Optional[dict[int, tuple]], optional): color of each class. Defaults to None.
        class_labels (Optional[dict[int, str]], optional): label of each class. Defaults to None.
        train_title (str, optional): title of train dataset. Defaults to "Train Dataset".
        test_title (str, optional): title of test dataset. Defaults to "Test Dataset".
        save_path (Optional[str | Path], optional): save path of graph. Defaults to None.
        **kwargs (Any): additional arguments for scatter plot.
    """
    if dataset.config.features is not None:
        feature_cols = dataset.config.features
    else:
        feature_cols = DEFAULT_FEATURE_COLS

    y_all = np.concatenate([dataset.y_train, dataset.y_test])
    if colors is None:
        colors = {
            int(class_value): color
            for class_value, color in zip(
                np.unique(y_all), sns.color_palette("viridis", n_colors=len(np.unique(y_all)))
            )
        }

    if class_labels is None:
        class_labels = {class_value: str(class_value) for class_value in np.unique(y_all)}

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.subplot(1, 2, 1)
    for class_value in np.unique(dataset.y_train):
        subset = dataset.X_train[np.where(dataset.y_train == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            color=colors.get(class_value),
            label=class_labels.get(class_value),
            **kwargs,
        )
        plt.xlabel(f"{feature_cols[0]}")
        plt.ylabel(f"{feature_cols[1]}")
        plt.legend(title="Class")
        plt.title(train_title)

    plt.subplot(1, 2, 2)
    for class_value in np.unique(dataset.y_test):
        subset = dataset.X_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            color=colors.get(class_value),
            label=class_labels.get(class_value),
            **kwargs,
        )
        plt.xlabel(f"{feature_cols[0]}")
        plt.ylabel(f"{feature_cols[1]}")
        plt.legend(title="Class")
        plt.title(test_title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qk_manager.datasets.schema import Dataset


def plot_dataset(
    dataset: Dataset,
    colors: dict[int, str] = {-1: "red", 1: "green"},
    class_labels: dict[int, str] = {-1: "-1", 1: "1"},
    save_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.subplot(1, 2, 1)
    for class_value in np.unique(dataset.y_train):
        subset = dataset.x_train[np.where(dataset.y_train == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors[class_value],
            label=class_labels[class_value],
        )
        plt.xlabel(f"{dataset.feature_cols[0]}")
        plt.ylabel(f"{dataset.feature_cols[1]}")
        plt.legend()
        plt.title("Train Dataset")

    plt.subplot(1, 2, 2)
    for class_value in np.unique(dataset.y_test):
        subset = dataset.x_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=colors[class_value],
            label=class_labels[class_value],
        )
        plt.xlabel(f"{dataset.feature_cols[0]}")
        plt.ylabel(f"{dataset.feature_cols[1]}")
        plt.legend()
        plt.title("Test Dataset")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

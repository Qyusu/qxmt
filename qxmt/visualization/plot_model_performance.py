from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qxmt.datasets.schema import Dataset
from qxmt.models.base import BaseKernelModel
from qxmt.visualization.utils import _create_class_labels, _create_colors


def plot_2d_predicted_result(
    dataset: Dataset,
    y_pred: np.ndarray,
    axis: list[int] = [0, 1],
    class_labels: Optional[dict[int, str]] = None,
    colors: Optional[dict[int, str]] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot predicted result on 2D plane.

    Args:
        dataset (Dataset): Dataset object. It contains test data.
        y_pred (np.ndarray): predicted labels.
        axis (list[int], optional): axis to plot (target feature col index). Defaults to [0, 1].
        class_labels (Optional[dict[int, str]], optional): label of each class. Defaults to None.
        colors (Optional[dict[int, str]], optional): color of each class. Defaults to None.
        save_path (Optional[str], optional): save path of graph. Defaults to None.
    """
    if dataset.config.features is not None:
        feature_cols = dataset.config.features
    else:
        feature_cols = [f"feature_{i}" for i in range(max(axis) + 1)]

    if colors is None:
        colors = _create_colors(dataset.y_test)

    if class_labels is None:
        class_labels = _create_class_labels(dataset.y_test)

    plt.figure(figsize=(7.5, 5), tight_layout=True)
    for class_value in np.unique(dataset.y_test):
        groud_subset = dataset.X_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            groud_subset[:, axis[0]],
            groud_subset[:, axis[1]],
            edgecolor=colors.get(class_value),
            facecolor="none",
            s=100,
            label=f"Groud Truth (label={class_labels.get(class_value)})",
        )

        predicted_subset = dataset.X_test[np.where(y_pred == class_value)]
        plt.scatter(
            predicted_subset[:, axis[0]],
            predicted_subset[:, axis[1]],
            marker="x",
            c=colors.get(class_value),
            label=f"Predicted (label={class_labels.get(class_value)})",
        )

    plt.xlabel(f"{feature_cols[axis[0]]}")
    plt.ylabel(f"{feature_cols[axis[1]]}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.title('"Groud Truth" VS "Predicted"')

    if save_path is not None:
        plt.savefig(save_path)


def plot_2d_decisionon_boundaries(
    model: BaseKernelModel,
    dataset: Dataset,
    axis: list[int] = [0, 1],
    step_size: float = 0.1,
    save_path: Optional[str | Path] = None,
) -> None:
    if dataset.config.features is not None:
        feature_cols = dataset.config.features
    else:
        feature_cols = [f"feature_{i}" for i in range(max(axis) + 1)]

    x_min, x_max = (
        dataset.X_test[:, axis[0]].min() - step_size,
        dataset.X_test[:, axis[0]].max() + step_size,
    )
    y_min, y_max = (
        dataset.X_test[:, axis[1]].min() - step_size,
        dataset.X_test[:, axis[1]].max() + step_size,
    )
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # calculate kernel matrix for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)

    # print(f"Calculating Kernel Matrix for {len(mesh_points)} mesh points")
    # K_mesh = model.get_kernel_matrix(mesh_points, dataset.X_train)
    # Z = model.predict(K_mesh)

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    if np.isnan(Z).any() or np.isinf(Z).any():
        Z = np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=-1.0)
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.5, levels=[-1, 0, 1])

    neg_subset = dataset.X_test[np.where(dataset.y_test == -1)]
    plt.scatter(neg_subset[:, 0], neg_subset[:, 1], marker="o", color="black")
    pos_subset = dataset.X_test[np.where(dataset.y_test == 1)]
    plt.scatter(pos_subset[:, 0], pos_subset[:, 1], marker="x", color="black")

    plt.xlabel(f"{feature_cols[0]}")
    plt.ylabel(f"{feature_cols[1]}")

    if save_path is not None:
        plt.savefig(save_path)

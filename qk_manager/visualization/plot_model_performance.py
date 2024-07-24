from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qk_manager.datasets.schema import Dataset
from qk_manager.models.base_kernel_model import BaseKernelModel


def plot_predicted_result(
    dataset: Dataset,
    y_pred: np.ndarray,
    colors: dict[int, str] = {-1: "red", 1: "green"},
    axis: list[int] = [0, 1],
    save_path: Optional[str] = None,
) -> None:

    plt.figure(figsize=(7, 5), tight_layout=True)
    for class_value in np.unique(dataset.y_test):
        groud_subset = dataset.x_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            groud_subset[:, axis[0]],
            groud_subset[:, axis[1]],
            edgecolor=colors[class_value],
            facecolor="none",
            s=100,
            label="Groud Truth",
        )

        predicted_subset = dataset.x_test[np.where(y_pred == class_value)]
        plt.scatter(
            predicted_subset[:, axis[0]],
            predicted_subset[:, axis[1]],
            marker="x",
            c=colors[class_value],
            label="Predicted",
        )

    plt.xlabel(f"{dataset.feature_cols[axis[0]]}")
    plt.ylabel(f"{dataset.feature_cols[axis[1]]}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
    plt.title('"Groud Truth" VS "Predicted"')

    if save_path is not None:
        plt.savefig(save_path)


def plot_decisionon_boundaries(
    model: BaseKernelModel,
    dataset: Dataset,
    # kernel_params: Optional[KernelParams] = None,
    step_size: float = 0.1,
    save_path: Optional[str] = None,
):
    x_min, x_max = (
        dataset.x_test[:, 0].min() - step_size,
        dataset.x_test[:, 0].max() + step_size,
    )
    y_min, y_max = (
        dataset.x_test[:, 1].min() - step_size,
        dataset.x_test[:, 1].max() + step_size,
    )
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # calculate kernel matrix for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)

    # if kernel_params is None:
    #     Z = model.predict(mesh_points)
    # else:
    #     print(f"Calculating Kernel Matrix for {len(mesh_points)} mesh points")
    #     K_mesh = compute_kernel(mesh_points, dataset.x_train, kernel_params)
    #     Z = model.predict(K_mesh)

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    if np.isnan(Z).any() or np.isinf(Z).any():
        Z = np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=-1.0)
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.5, levels=[-1, 0, 1])

    neg_subset = dataset.x_test[np.where(dataset.y_test == -1)]
    plt.scatter(neg_subset[:, 0], neg_subset[:, 1], marker="o", color="black")
    pos_subset = dataset.x_test[np.where(dataset.y_test == 1)]
    plt.scatter(pos_subset[:, 0], pos_subset[:, 1], marker="x", color="black")

    plt.xlabel(f"{dataset.feature_cols[0]}")
    plt.ylabel(f"{dataset.feature_cols[1]}")

    if save_path is not None:
        plt.savefig(save_path)

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qxmt.constants import DEFAULT_COLOR_MAP
from qxmt.decorators import notify_long_running
from qxmt.models.qkernels import QSVC


def plot_2d_predicted_result(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: Optional[list[str]] = None,
    axis: list[int] = [0, 1],
    title: str = '"Groud Truth" VS "Predicted"',
    truth_label: str = "Ground Truth",
    pred_label: str = "Predicted",
    colors: Optional[dict[int, tuple[float, float, float]]] = None,
    class_labels: Optional[dict[int, str]] = None,
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot predicted result on 2D plane.

    Args:
        X (np.ndarray): feature values.
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        feature_cols (Optional[list[str]], optional): feature column names. Defaults to None.
        axis (list[int], optional): axis to plot (target feature col index). Defaults to [0, 1].
        title (str, optional): title of the plot. Defaults to '"Groud Truth" VS "Predicted"'.
        truth_label (str, optional): label of ground truth. Defaults to "Ground Truth".
        pred_label (str, optional): label of predicted. Defaults to "Predicted".
        colors (Optional[dict[int, tuple[float, float, float]]], optional): color of each class. Defaults to None.
        class_labels (Optional[dict[int, str]], optional): label of each class. Defaults to None.
        save_path (Optional[str | Path], optional): save path of graph. Defaults to None.
        **kwargs (Any): additional arguments for scatter plot.
    """
    assert X.shape[0] == len(y_true) == len(y_pred), "Length of X , y_true and y_pred must be the same."

    if feature_cols is None:
        feature_cols = [f"feature_{i}" for i in range(max(axis) + 1)]

    # set each class color and label
    colors = (
        colors
        if colors is not None
        else {
            i: (r, g, b)
            for i, (r, g, b) in enumerate(sns.color_palette(DEFAULT_COLOR_MAP, n_colors=len(np.unique(y_true))))
        }
    )
    class_labels = (
        class_labels
        if class_labels is not None
        else {class_value: str(class_value) for class_value in np.unique(y_true)}
    )

    # plot grpoed truth and predicted result
    plt.figure(figsize=(7, 5), tight_layout=True)
    color_labels = []
    for i, class_value in enumerate(np.unique(y_true)):
        groud_subset = X[np.where(y_true == class_value)]
        plt.scatter(
            groud_subset[:, axis[0]],
            groud_subset[:, axis[1]],
            edgecolor=colors[i],
            facecolor="none",
            s=100,
            **kwargs,
        )

        predicted_subset = X[np.where(y_pred == class_value)]
        plt.scatter(
            predicted_subset[:, axis[0]],
            predicted_subset[:, axis[1]],
            marker="x",
            color=colors[i],
            **kwargs,
        )

        # legend for colors, it is empty because we want to show only labels
        color_label = plt.scatter([], [], marker="o", color=colors[i], label=f"{class_labels.get(class_value)}")
        color_labels.append(color_label)

    # legend for markers, it is empty because we want to show only labels
    truth_label_handle = plt.scatter([], [], label=truth_label, marker="o", facecolor="none", color="black")
    predict_label_handle = plt.scatter([], [], label=pred_label, marker="x", color="black")
    marker_legend = plt.legend(
        title="marker type",
        handles=[truth_label_handle, predict_label_handle],
        loc="upper right",
        bbox_to_anchor=(1.4, 1),
    )
    _ = plt.legend(title="Class", handles=color_labels, loc="upper right", bbox_to_anchor=(1.4, 0.8))

    plt.gca().add_artist(marker_legend)

    plt.xlabel(f"{feature_cols[axis[0]]}")
    plt.ylabel(f"{feature_cols[axis[1]]}")
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()


@notify_long_running
def plot_2d_decision_boundaries(
    model: QSVC,
    X: np.ndarray,
    y: np.ndarray,
    grid_resolution: int = 100,
    support_vectors: bool = True,
    feature_cols: Optional[list[str]] = None,
    title: str = "Decision boundaries of QSVC",
    cmap: str = DEFAULT_COLOR_MAP,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot decision boundaries of QSVC on 2D plane.

    Args:
        model (QSVC): QSVC model.
        X (np.ndarray): Feature values of the data.
        y (np.ndarray): Labels of the data.
        grid_resolution (int, optional): Resolution of the grid. Defaults to 100.
        support_vectors (bool, optional): Plot support vectors or not. Defaults to True.
        feature_cols (Optional[list[str]], optional): Feature column names. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Decision boundaries of QSVC".
        cmap (str, optional): Color map for the plot. Defaults to DEFAULT_COLOR_MAP.
        save_path (Optional[str | Path], optional): Save path of the graph. Defaults to None.
    """
    assert X.shape[0] == len(y), "Length of X and y must be the same."

    if X.shape[1] != 2:
        raise ValueError("This function only supports 2D datasets.")

    if feature_cols is None:
        feature_cols = ["feature_0", "feature_1"]

    _, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    margin = 1
    ax.set(xlim=(x_min - margin, x_max + margin), ylim=(y_min - margin, y_max + margin))

    # Generate grid
    xx, yy = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, grid_resolution),
        np.linspace(y_min - margin, y_max + margin, grid_resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Shape (n_grid_points, 2)

    # Compute predictions over the grid
    Z = model.predict(grid_points)
    is_multi_class = len(model.classes_) > 2
    if not is_multi_class:
        decision_function = model.decision_function(grid_points)

    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    ax.pcolormesh(xx, yy, Z, cmap=cmap, shading="auto", alpha=0.3, vmin=y.min(), vmax=y.max())

    if not is_multi_class:
        # Plot decision boundary and margins
        Z_decision = decision_function.reshape(xx.shape)
        ax.contour(
            xx,
            yy,
            Z_decision,
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
        )

    if support_vectors and hasattr(model, "support_"):
        # Plot support vectors
        ax.scatter(
            X[model.support_, 0],
            X[model.support_, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k", cmap=cmap, vmin=y.min(), vmax=y.max())
    plt.xlabel(f"{feature_cols[0]}")
    plt.ylabel(f"{feature_cols[1]}")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Class", bbox_to_anchor=(1.2, 1))
    ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

from pathlib import Path
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

from qxmt.datasets.schema import Dataset
from qxmt.decorators import notify_long_running
from qxmt.models.qsvm import QSVM
from qxmt.visualization.graph_settings import _create_class_labels, _create_colors


def plot_2d_predicted_result(
    dataset: Dataset,
    y_pred: np.ndarray,
    axis: list[int] = [0, 1],
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot predicted result on 2D plane.

    Args:
        dataset (Dataset): Dataset object. It contains test data.
        y_pred (np.ndarray): predicted labels.
        axis (list[int], optional): axis to plot (target feature col index). Defaults to [0, 1].
        save_path (Optional[str], optional): save path of graph. Defaults to None.
        **kwargs (Any): additional arguments for plot.
            truth_label (str, optional): label of ground truth. Defaults to "Ground Truth".
            pred_label (str, optional): label of predicted. Defaults to "Predicted".
            title (str, optional): title of the plot. Defaults to '"Groud Truth" VS "Predicted"'.
            colors (Optional[dict[int, str]], optional): color of each class. Defaults to None.
            class_labels (Optional[dict[int, str]], optional): label of each class. Defaults to None.
    """
    if dataset.config.features is not None:
        feature_cols = dataset.config.features
    else:
        feature_cols = [f"feature_{i}" for i in range(max(axis) + 1)]

    colors = cast(dict[int, str] | None, kwargs.get("colors", None))
    if colors is None:
        colors = _create_colors(dataset.y_test)

    class_labels = cast(dict[int, str] | None, kwargs.get("class_labels", None))
    if class_labels is None:
        class_labels = _create_class_labels(dataset.y_test)

    plt.figure(figsize=(7, 5), tight_layout=True)
    color_labels = []
    for class_value in np.unique(dataset.y_test):
        groud_subset = dataset.X_test[np.where(dataset.y_test == class_value)]
        plt.scatter(
            groud_subset[:, axis[0]],
            groud_subset[:, axis[1]],
            edgecolor=colors.get(class_value),
            facecolor="none",
            s=100,
        )

        predicted_subset = dataset.X_test[np.where(y_pred == class_value)]
        plt.scatter(
            predicted_subset[:, axis[0]],
            predicted_subset[:, axis[1]],
            marker="x",
            c=colors.get(class_value),
        )

        # legend for colors, it is empty because we want to show only labels
        color_label = plt.scatter(
            [], [], marker="o", c=colors.get(class_value), label=f"{class_labels.get(class_value)}"
        )
        color_labels.append(color_label)

    # legend for markers, it is empty because we want to show only labels
    truth_label = plt.scatter(
        [], [], label=str(kwargs.get("truth_label", "Ground Truth")), marker="o", facecolor="none", color="black"
    )
    predict_label = plt.scatter([], [], label=str(kwargs.get("pred_label", "Predicted")), marker="x", color="black")
    marker_legend = plt.legend(
        title="marker type", handles=[truth_label, predict_label], loc="upper right", bbox_to_anchor=(1.4, 1)
    )
    _ = plt.legend(title="Class", handles=color_labels, loc="upper right", bbox_to_anchor=(1.4, 0.8))

    plt.gca().add_artist(marker_legend)

    plt.xlabel(f"{feature_cols[axis[0]]}")
    plt.ylabel(f"{feature_cols[axis[1]]}")

    title = str(kwargs.get("title", '"Groud Truth" VS "Predicted"'))
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()


@notify_long_running
def plot_2d_decisionon_boundaries(
    model: QSVM,
    dataset: Dataset,
    grid_resolution: int = 10,
    support_vectors: bool = True,
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot decision boundaries of QSVM on 2D plane.

    Args:
        model (QSVM): QSVM model.
        dataset (Dataset): Dataset object. It contains train data.
        grid_resolution (int, optional): resolution of grid. Defaults to 10.
        support_vectors (bool, optional): plot support vectors or not. Defaults to True.
        save_path (Optional[str  |  Path], optional): save path of graph. Defaults to None.
        **kwargs (Any): additional arguments for plot.
            title (str, optional): title of the plot. Defaults to "Decision boundaries of QSVC".
    """
    X, y = dataset.X_train, dataset.y_train
    is_multi_class = len(np.unique(y)) > 2

    if X.shape[1] != 2:
        raise ValueError("This function only supports 2D dataset.")

    if dataset.config.features is not None:
        feature_cols = dataset.config.features
    else:
        feature_cols = ["feature_0", "feature_1"]

    _, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    x_min, x_max, y_min, y_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
    margin = 1
    ax.set(xlim=(x_min - margin, x_max + margin), ylim=(y_min - margin, y_max + margin))

    # Plot decision boundary and margins
    common_params = {"estimator": model.model, "X": X, "grid_resolution": grid_resolution, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )

    if not is_multi_class:
        # Only plot decision boundary and margins when it is binary classification
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
        )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            X[model.model.support_, 0],
            X[model.model.support_, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    plt.xlabel(f"{feature_cols[0]}")
    plt.ylabel(f"{feature_cols[1]}")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Class", bbox_to_anchor=(1.2, 1))

    title = str(kwargs.get("title", "Decision boundaries of QSVC"))
    ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

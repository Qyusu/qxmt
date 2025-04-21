from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid: bool = True,
    title: str = "Residual Plot",
    x_label: str = "Predicted Values",
    y_label: str = "Residuals",
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot residuals vs predicted values.
    residuals caluculated by y_true - y_pred.

    Args:
        y_true (np.ndarray): actual values of the data.
        y_pred (np.ndarray): predicted values of the data.
        grid (bool, optional): grid on the plot. Defaults to True.
        title (str, optional): title of the plot. Defaults to "Residual Plot".
        x_label (str, optional): label of the x-axis. Defaults to "Predicted Values".
        y_label (str, optional): label of the y-axis. Defaults to "Residuals".
        save_path (Optional[str  |  Path], optional): save path of graph. Defaults to None.
        **kwargs (Any): additional arguments for plot.
    """
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
    residuals = y_true - y_pred

    # Residuals vs Predicted values
    plt.figure(figsize=(7, 5), tight_layout=True)
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors="k", color="blue")
    plt.axhline(y=0, color="red", linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid: bool = True,
    title: str = "Actual vs Predicted Plot",
    x_label: str = "Actual Values",
    y_label: str = "Predicted Values",
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot actual vs predicted values.

    Args:
        y_true (np.ndarray): actual values of the data.
        y_pred (np.ndarray): predicted values of the data.
        grid (bool, optional): grid on the plot. Defaults to True.
        title (str, optional): title of the plot. Defaults to "Actual vs Predicted Plot".
        x_label (str, optional): label of the x-axis. Defaults to "Actual Values".
        y_label (str, optional): label of the y-axis. Defaults to "Predicted Values".
        save_path (Optional[str  |  Path], optional): save path of graph. Defaults to None.
        **kwargs (Any): additional arguments for plot.
    """
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."

    plt.figure(figsize=(7, 5), tight_layout=True)

    # Scatter plot of actual vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", color="blue")
    # Line of perfect predictions
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--", lw=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

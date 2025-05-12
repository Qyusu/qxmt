from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


def plot_optimization_history(
    cost_history: list[float],
    baseline_cost: Optional[float] = None,
    cost_label: Optional[str] = None,
    baseline_label: Optional[str] = None,
    marker: str = "o",
    markersize: int = 5,
    title: str = "Optimization History",
    step_num: int = 10,
    x_label: str = "Step",
    y_label: str = "Cost",
    grid: bool = False,
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot the optimization history.

    Args:
        cost_history (list[float]): The cost history of each optimization step.
        baseline_cost (float): The baseline cost of the optimization.
        cost_label (Optional[str], optional): The label of the cost history. Defaults to None.
        baseline_label (Optional[str], optional): The label of the baseline cost. Defaults to None.
        marker (str, optional): The marker of the cost history. Defaults to "o".
        markersize (int, optional): The size of the marker. Defaults to 5.
        title (str, optional): The title of the plot. Defaults to "Optimization History".
        step_num (int, optional): The number of steps to show on the x-axis. Defaults to 10.
        x_label (str, optional): The label of the x-axis. Defaults to "Step".
        y_label (str, optional): The label of the y-axis. Defaults to "Cost".
        save_path (Optional[str | Path], optional): The path to save the plot. Defaults to None.
        **kwargs (Any): Additional arguments for the plot method.
    """
    step_num = len(cost_history)
    plt.figure(figsize=(7, 5), tight_layout=True)
    plt.plot(range(step_num), cost_history, marker=marker, markersize=markersize, label=cost_label, **kwargs)
    if baseline_cost is not None:
        plt.axhline(y=baseline_cost, color="black", linestyle="--", label=baseline_label)
    x_tick_interval = max(1, step_num // 10)
    x_ticks = range(0, step_num, x_tick_interval)
    plt.xticks(x_ticks)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(grid)

    if cost_label is not None or baseline_label is not None:
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

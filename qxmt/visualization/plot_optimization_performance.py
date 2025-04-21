from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt


def plot_optimization_history(
    cost_history: list[float],
    marker: str = "o",
    title: Optional[str] = None,
    x_tick_interval: int = 5,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot the optimization history.

    Args:
        cost_history (list[float]): The cost history of each optimization step.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        x_tick_interval (int, optional): The interval of the x-axis ticks. Defaults to 5.
        x_label (Optional[str], optional): The label of the x-axis. Defaults to None.
        y_label (Optional[str], optional): The label of the y-axis. Defaults to None.
        save_path (Optional[str | Path], optional): The path to save the plot. Defaults to None.
        **kwargs (Any): Additional arguments for the plot method.
    """
    step_num = len(cost_history)
    plt.figure(figsize=(7, 5), tight_layout=True)
    plt.plot(range(step_num), cost_history, marker=marker, **kwargs)

    x_ticks = range(0, step_num, x_tick_interval)
    plt.xticks(x_ticks)

    x_label = x_label if x_label is not None else "Step"
    y_label = y_label if y_label is not None else "Cost"
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    title = title if title is not None else f"Optimization History (Total Steps: {step_num})"
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

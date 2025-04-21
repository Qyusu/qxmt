from logging import Logger
from pathlib import Path
from typing import Any, Literal, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from qxmt.constants import DEFAULT_COLOR_MAP
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)
RUN_ID_COL = "run_id"


def _check_existence_of_metrics(df: pd.DataFrame, metrics: list[str], logger: Logger = LOGGER) -> list[str]:
    """Check if the target metrics exist in the DataFrame columns.

    Args:
        df (pd.DataFrame): dataframe that contains the calculated metrics
        metrics (list[str]): target metric names
        logger (Logger, optional): logger object. Defaults to LOGGER.

    Returns:
        list[str]: valid metric names
    """
    valid_metrics = []
    for metric in metrics:
        if metric not in df.columns:
            # raise ValueError(f"{metric} is not in the DataFrame columns.")
            logger.warning(f'"{metric}" is not in the DataFrame columns. skip this metric.')
        else:
            valid_metrics.append(metric)

    return valid_metrics


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    chart_type: Literal["bar", "line"] = "bar",
    run_id_col: str = RUN_ID_COL,
    run_ids: Optional[list[int]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    grid: Optional[bool] = True,
    title: Optional[str] = None,
    color: Optional[tuple[float, float, float]] = None,
    save_path: Optional[Path] = None,
    logger: Logger = LOGGER,
    **kwargs: Any,
) -> None:
    """Plot bar chart of the target metric. x-axis is run_id.

    Args:
        df (pd.DataFrame): dataframe that contains the calculated metrics
        metric (str): target metric name
        chart_type (Literal["bar", "line"], optional): chart type ("bar" or "line"). Defaults to "bar".
        run_id_col (str, optional): column name of run_id. Defaults to RUN_ID_COL.
        run_ids (Optional[list[int]], optional): run_ids to plot. Defaults to None.
        xlabel (Optional[str], optional): x-axis label. Defaults to "run_id".
        ylabel (Optional[str], optional): y-axis label. Defaults to f'"{valid_metric}" score'.
        ylim (Optional[tuple[float, float]], optional): y-axis limit. Defaults to None.
        grid (Optional[bool], optional): grid option. Defaults to True.
        title (Optional[str], optional): title of the plot. Defaults to None.
        color (Optional[tuple[float, float, float]], optional): color of the plot. Defaults to None.
        save_path (Optional[Path], optional): save path for the plot. Defaults to None.
        logger (Logger, optional): logger object. Defaults to LOGGER.
        **kwargs (Any): additional arguments for plot.
    """
    if run_ids is not None:
        df = df[df[run_id_col].isin(run_ids)]

    valid_metric = _check_existence_of_metrics(df, [metric], logger)[0]

    plt.figure(figsize=(10, 6), tight_layout=True)
    x = [i for i in range(len(df))]
    color = color if color is not None else sns.color_palette(DEFAULT_COLOR_MAP)[0]
    if chart_type == "bar":
        plt.bar(x, df[valid_metric], color=color, **kwargs)
    elif chart_type == "line":
        plt.plot(x, df[valid_metric], color=color, marker="o", **kwargs)
    else:
        raise ValueError(f"chart_type '{chart_type}' is not supported.")
    plt.xlabel(xlabel if xlabel is not None else run_id_col)
    plt.ylabel(ylabel if ylabel is not None else f'"{valid_metric}" score')
    plt.xticks(x, list(df[run_id_col]))

    ylim = ylim if ylim is not None else (0.0, df[valid_metric].max() * 1.2)
    plt.ylim(*ylim)

    if chart_type == "line":
        plt.grid(grid if grid is not None else True)

    plt.title(title if title is not None else f'"{valid_metric}" score')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()


def plot_metrics_side_by_side(
    df: pd.DataFrame,
    metrics: list[str],
    chart_type: Literal["bar", "line"] = "bar",
    run_id_col: str = RUN_ID_COL,
    run_ids: Optional[list[int]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    grid: Optional[bool] = True,
    title: Optional[str] = None,
    colors: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
    logger: Logger = LOGGER,
    **kwargs: Any,
) -> None:
    """Plot bar chart of the target metrics side by side. x-axis is run_id.

    Args:
        df (pd.DataFrame): dataframe that contains the calculated metrics
        metrics (list[str]): target metric names
        chart_type (Literal["bar", "line"], optional): chart type ("bar" or "line"). Defaults to "bar".
        run_id_col (str, optional): column name of run_id. Defaults to RUN_ID_COL.
        run_ids (Optional[list[int]], optional): run_ids to plot. Defaults to None.
        xlabel (Optional[str], optional): x-axis label. Defaults to "run_id".
        ylabel (Optional[str], optional): y-axis label. Defaults to "metrics score".
        ylim (Optional[tuple[float, float]], optional): y-axis limit. Defaults to None.
        grid (Optional[bool], optional): grid option. Defaults to True.
        title (Optional[str], optional): title of the plot. Defaults to None.
        colors (Optional[list[str]], optional): color of the plot. Defaults to None.
        save_path (Optional[Path], optional): save path for the plot. Defaults to None.
        logger (Logger, optional): logger object. Defaults to LOGGER.
        **kwargs (Any): additional arguments for plot.
    """
    if run_ids is not None:
        df = df[df[run_id_col].isin(run_ids)]

    valid_metrics = _check_existence_of_metrics(df, metrics, logger)

    plt.figure(figsize=(10, 6), tight_layout=True)
    width = 1.0 / (len(valid_metrics) + 1)
    color_palette: list = cast(list, colors if colors is not None else sns.color_palette(DEFAULT_COLOR_MAP))
    y_max = 0.0
    for i, metric in enumerate(valid_metrics):
        if chart_type == "bar":
            x = [j + i * width for j in range(len(df))]
            plt.bar(x, df[metric], width=width, label=metric, color=color_palette[i], **kwargs)
        elif chart_type == "line":
            x = [j for j in range(len(df))]
            plt.plot(x, df[metric], label=metric, color=color_palette[i], marker="o", **kwargs)
        else:
            raise ValueError(f"chart_type '{chart_type}' is not supported.")
        y_max = max(y_max, max(df[metric]))

    plt.ylabel(ylabel if ylabel is not None else "metrics score")
    plt.xlabel(xlabel if xlabel is not None else run_id_col)
    if chart_type == "bar":
        x_ticks = [i + width * 0.5 * (len(valid_metrics) - 1) for i in range(len(df))]
    else:
        x_ticks = [i for i in range(len(df))]
    plt.xticks(x_ticks, list(df[run_id_col]))

    ylim = ylim if ylim is not None else (0.0, y_max * 1.2)
    plt.ylim(*ylim)

    plt.legend(loc="upper right")

    if chart_type == "line":
        plt.grid(grid if grid is not None else True)

    plt.title(title if title is not None else "metrics score")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

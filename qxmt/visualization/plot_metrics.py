from logging import Logger
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

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
    run_ids: Optional[list[int]] = None,
    save_path: Optional[Path] = None,
    logger: Logger = LOGGER,
) -> None:
    """Plot bar chart of the target metric. x-axis is run_id.

    Args:
        df (pd.DataFrame): dataframe that contains the calculated metrics
        metric (str): target metric name
        run_ids (Optional[list[int]], optional): run_ids to plot. Defaults to None.
        save_path (Optional[Path], optional): save path for the plot. Defaults to None.
        logger (Logger, optional): logger object. Defaults to LOGGER.
    """
    if run_ids is not None:
        df = df[df[RUN_ID_COL].isin(run_ids)]

    valid_metric = _check_existence_of_metrics(df, [metric], logger)[0]

    plt.figure(figsize=(10, 6), tight_layout=True)
    x = [i for i in range(len(df))]
    plt.bar(x, df[valid_metric])
    plt.ylim(0, 1.05)
    plt.ylabel(f'"{valid_metric}" score')
    plt.xlabel("run_id")
    plt.xticks(x, list(df["run_id"]))

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_metrics_side_by_side(
    df: pd.DataFrame,
    metrics: list[str],
    run_ids: Optional[list[int]] = None,
    save_path: Optional[Path] = None,
    logger: Logger = LOGGER,
) -> None:
    """Plot bar chart of the target metrics side by side. x-axis is run_id.

    Args:
        df (pd.DataFrame): dataframe that contains the calculated metrics
        metrics (list[str]): target metric names
        run_ids (Optional[list[int]], optional): run_ids to plot. Defaults to None.
        save_path (Optional[Path], optional): save path for the plot. Defaults to None.
        logger (Logger, optional): logger object. Defaults to LOGGER.
    """
    if run_ids is not None:
        df = df[df[RUN_ID_COL].isin(run_ids)]

    valid_metrics = _check_existence_of_metrics(df, metrics, logger)

    plt.figure(figsize=(10, 6), tight_layout=True)
    width = 1.0 / (len(valid_metrics) + 1)
    for i, metric in enumerate(valid_metrics):
        x = [j + i * width for j in range(len(df))]
        plt.bar(x, df[metric], width=width, label=metric)

    plt.ylim(0, 1.3)
    plt.ylabel("metrics score")
    plt.xlabel("run_id")
    x_ticks = [i + width * 0.5 * (len(valid_metrics) - 1) for i in range(len(df))]
    plt.xticks(x_ticks, list(df["run_id"]))
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

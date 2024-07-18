from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _check_existence_of_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    valid_metrics = []
    for metric in metrics:
        if metric not in df.columns:
            # raise ValueError(f"{metric} is not in the DataFrame columns.")
            print(f"{metric} is not in the DataFrame columns. skip this metric.")
        else:
            valid_metrics.append(metric)

    return valid_metrics


def plot_metric(df: pd.DataFrame, metric: str, save_path: Optional[Path] = None) -> None:
    valid_metric = _check_existence_of_metrics(df, [metric])[0]
    plt.figure(figsize=(10, 6))
    x = [i for i in range(len(df))]
    plt.bar(x, df[valid_metric])

    plt.ylim(0, 1.05)
    plt.ylabel(f'"{valid_metric}" score')
    plt.xlabel("run_id")
    plt.xticks(x, list(df["run_id"]))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_metrics_side_by_side(df: pd.DataFrame, metrics: list[str], save_path: Optional[Path] = None) -> None:
    valid_metrics = _check_existence_of_metrics(df, metrics)

    plt.figure(figsize=(10, 6))
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
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

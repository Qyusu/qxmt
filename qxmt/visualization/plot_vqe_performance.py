from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from qxmt.constants import DEFAULT_COLOR_MAP


def plot_energy_difference_by_bond_length(
    energy_list: list[float],
    baseline_energy_list: list[float],
    bond_length_list: list[float],
    absolute: bool = False,
    title: str = "Energy Difference by Bond Length",
    x_label: str = "Bond Length",
    y_label: str = "Energy Difference",
    grid: bool = True,
    x_ticks_points: int = 10,
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot the energy difference between the VQE and the baseline energy.

    Args:
        cost_history (list[float]): The cost history of the VQE.
        baseline_cost (float): The baseline cost of the VQE.
        bond_length_list (list[float]): The bond length list.
        absolute (bool, optional): Whether to use the absolute value of the energy difference. Defaults to False.
        title (str, optional): The title of the plot. Defaults to "Energy Difference by Bond Length".
        x_label (str, optional): The label of the x-axis. Defaults to "Bond Length".
        y_label (str, optional): The label of the y-axis. Defaults to "Energy Difference".
        grid (bool, optional): Whether to show the grid. Defaults to True.
        x_ticks_points (int, optional): The number of points on the x-axis. Defaults to 10.
        save_path (Optional[str | Path], optional): The path to save the plot. Defaults to None.
    """
    if len(energy_list) != len(baseline_energy_list) != len(bond_length_list):
        raise ValueError(
            "The length of the energy list, the baseline energy list, and the bond length list must be the same."
        )

    if absolute:
        energy_difference = [
            abs(energy - baseline_energy) for energy, baseline_energy in zip(energy_list, baseline_energy_list)
        ]
    else:
        energy_difference = [
            energy - baseline_energy for energy, baseline_energy in zip(energy_list, baseline_energy_list)
        ]

    plt.figure(figsize=(7, 5), tight_layout=True)
    color = sns.color_palette(DEFAULT_COLOR_MAP)[0]
    plt.plot(bond_length_list, energy_difference, color=color, **kwargs)

    # dynamic x-axis tick
    num_points = len(bond_length_list)
    max_ticks = min(x_ticks_points, num_points)
    plt.gca().xaxis.set_major_locator(MaxNLocator(max_ticks))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()


def plot_pec(
    energy_list: list[float],
    hf_energy_list: list[float],
    fci_energy_list: list[float],
    bond_length_list: list[float],
    title: str = "Potential Energy Curve",
    x_label: str = "Bond Length",
    y_label: str = "Energy",
    hf_label: str = "HF",
    fci_label: str = "FCI",
    vqe_label: str = "VQE",
    grid: bool = True,
    x_ticks_points: int = 10,
    save_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> None:
    """Plot the Potential Energy Curve (PEC) of the VQE.

    Args:
        energy_list (list[float]): The energy list.
        hf_energy_list (list[float]): The HF energy list.
        fci_energy_list (list[float]): The FCI energy list.
        bond_length_list (list[float]): The bond length list.
        title (str, optional): The title of the plot. Defaults to "Potential Energy Curve".
        x_label (str, optional): The label of the x-axis. Defaults to "Bond Length".
        y_label (str, optional): The label of the y-axis. Defaults to "Energy".
        hf_label (str, optional): The label of the HF energy. Defaults to "HF".
        fci_label (str, optional): The label of the FCI energy. Defaults to "FCI".
        vqe_label (str, optional): The label of the VQE energy. Defaults to "VQE".
        grid (bool, optional): Whether to show the grid. Defaults to True.
        x_ticks_points (int, optional): The number of points on the x-axis. Defaults to 10.
        save_path (Optional[str | Path], optional): The path to save the plot. Defaults to None.
    """
    if len(energy_list) != len(hf_energy_list) != len(fci_energy_list) != len(bond_length_list):
        raise ValueError(
            """The length of the energy list, the HF energy list, the FCI energy list,
            and the bond length list must be the same."""
        )

    plt.figure(figsize=(7, 5), tight_layout=True)
    colors = sns.color_palette(DEFAULT_COLOR_MAP)
    plt.plot(bond_length_list, hf_energy_list, linestyle="--", color=colors[0], label=hf_label, **kwargs)
    plt.plot(bond_length_list, fci_energy_list, linestyle="-", color=colors[1], label=fci_label, **kwargs)
    plt.scatter(bond_length_list, energy_list, marker="o", color=colors[2], label=vqe_label, **kwargs)

    # dynamic x-axis tick
    num_points = len(bond_length_list)
    max_ticks = min(x_ticks_points, num_points)
    plt.gca().xaxis.set_major_locator(MaxNLocator(max_ticks))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(grid)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

from qxmt.visualization import __all__

EXPECTED_ALL = [
    "plot_2d_decision_boundaries",
    "plot_2d_predicted_result",
    "plot_2d_dataset",
    "plot_metric",
    "plot_metrics_side_by_side",
    "plot_optimization_history",
    "plot_residual",
    "plot_actual_vs_predicted",
    "plot_energy_difference_by_bond_length",
    "plot_pec",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

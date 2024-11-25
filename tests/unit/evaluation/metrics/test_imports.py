from qxmt.evaluation.metrics import __all__

EXPECTED_ALL = [
    "BaseMetric",
    "Accuracy",
    "Recall",
    "Precision",
    "F1Score",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "R2Score",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

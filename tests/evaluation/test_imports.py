from qxmt.evaluation import __all__

EXPECTED_ALL = [
    "BaseMetric",
    "Accuracy",
    "Recall",
    "Precision",
    "F1Score",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "R2Score",
    "Evaluation",
    "ClassificationEvaluation",
    "RegressionEvaluation",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

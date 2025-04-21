from qxmt.evaluation import __all__

EXPECTED_ALL = [
    "Evaluation",
    "ClassificationEvaluation",
    "RegressionEvaluation",
    "VQEEvaluation",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

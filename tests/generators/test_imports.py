import os

from qxmt.generators import __all__

if os.getenv("USE_LLM", "FALSE").lower() == "true":
    EXPECTED_ALL = ["DescriptionGenerator"]
else:
    EXPECTED_ALL = []


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

import importlib

import pytest


@pytest.mark.parametrize(
    "use_llm, expected_all",
    [
        ("false", []),
        ("true", ["DescriptionGenerator"]),
    ],
)
def test_all_imports(monkeypatch: pytest.MonkeyPatch, use_llm: str, expected_all: list[str]) -> None:
    monkeypatch.setenv("USE_LLM", use_llm)

    import qxmt.generators

    importlib.reload(qxmt.generators)
    from qxmt.generators import __all__

    assert set(__all__) == set(expected_all)

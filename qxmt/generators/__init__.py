import os

if os.getenv("USE_LLM", "FALSE").lower() == "true":
    from qxmt.generators.description import DescriptionGenerator

    __all__ = [
        "DescriptionGenerator",
    ]
else:
    __all__ = []

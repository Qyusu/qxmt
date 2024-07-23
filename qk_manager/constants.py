from pathlib import Path

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODULE_SRC: Path = Path(__file__).resolve().parents[0]

DEFAULT_EXP_DIRC: Path = MODULE_HOME / "experiments"
DEFAULT_EXP_DB_FILE: Path = Path("experiment.json")

DEFAULT_METRICS_NAME: list[str] = ["accuracy", "precision", "recall", "f1_score"]

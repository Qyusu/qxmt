from pathlib import Path
from typing import Any

import pennylane as qml
import pytz

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODULE_SRC: Path = Path(__file__).resolve().parents[0]

DEFAULT_EXP_DIRC: Path = MODULE_HOME / "experiments"
DEFAULT_EXP_DB_FILE: Path = Path("experiment.json")

SUPPORTED_PLATFORMS: list[str] = ["pennylane"]
PENNYLANE_DEVICES: tuple[Any, ...] = (qml.devices.Device, qml.Device, qml.QubitDevice)

DEFAULT_MODEL_NAME: str = "model.pkl"

DEFAULT_METRICS_NAME: list[str] = ["accuracy", "precision", "recall", "f1_score"]

TZ = pytz.timezone("Asia/Tokyo")

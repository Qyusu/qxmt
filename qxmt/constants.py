import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pennylane as qml
import pytz

# set module path for developer environment
MODULE_HOME: Path = Path(__file__).resolve().parents[1]

# set current working directory for user environment
CURRENT_WORKING_DIR: Path = Path.cwd()

# set project root directory.
# it swith by environment variable "QXMT_ENV"
PROJECT_ROOT_DIR: Path = (
    MODULE_HOME if os.getenv("QXMT_ENV", "").lower() == "dev" else CURRENT_WORKING_DIR.resolve().parents[0]
)

# set default experiment directory.
# it swith by environment variable "QXMT_ENV"
DEFAULT_EXP_DIRC: Path = PROJECT_ROOT_DIR / "experiments"

# set default experiment file name.
DEFAULT_EXP_DB_FILE: Path = Path("experiment.json")

# set supported quantum platforms and devices
SUPPORTED_PLATFORMS: list[str] = ["pennylane"]
PENNYLANE_DEVICES: tuple[Any, ...] = (qml.devices.Device, qml.Device, qml.QubitDevice)

# set default model name
DEFAULT_MODEL_NAME: str = "model.pkl"

# set default shot results file name
DEFAULT_SHOT_RESULTS_NAME: str = "shots.h5"

# set default LLM for generating description
LLM_MODEL_PATH = "microsoft/Phi-3-mini-128k-instruct"

# set default n_jobs for parallel processing
# it mainly used in kernel calculation, cross validation
DEFAULT_N_JOBS = max(1, int(mp.cpu_count() * 0.6))

# set default timezone
TZ: pytz.BaseTzInfo = pytz.timezone("Asia/Tokyo")

# set color mapt for visualization
DEFAULT_COLOR_MAP = "viridis"

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

# set default config file name.
DEFAULT_EXP_CONFIG_FILE: Path = Path("config.yaml")

# set alias for environment variable name
IBMQ_API_KEY: str = "IBMQ_API_KEY"
AWS_ACCESS_KEY_ID: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_DEFAULT_REGION: str = "AWS_DEFAULT_REGION"

# set supported quantum platforms and devices
PENNYLANE_PLATFORM: str = "pennylane"
SUPPORTED_PLATFORMS: list[str] = [PENNYLANE_PLATFORM]
_pl_devices: list[Any] = []
if hasattr(qml.devices, "Device"):
    _pl_devices.append(qml.devices.Device)
if hasattr(qml.devices, "LegacyDevice"):
    _pl_devices.append(getattr(qml.devices, "LegacyDevice"))
if hasattr(qml.devices, "QubitDevice"):
    _pl_devices.append(getattr(qml.devices, "QubitDevice"))
PENNYLANE_DEVICES: tuple[Any, ...] = tuple(_pl_devices) if _pl_devices else tuple()

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

import re
from importlib.metadata import entry_points
from typing import Any

Devices = None
AMAZON_PROVIDER_NAME = "Amazon_Braket"
AMAZON_BRAKET_LOCAL_DEVICES = ["braket.local.qubit"]
AMAZON_BRAKET_REMOTE_DEVICES = ["braket.aws.qubit"]
AMAZON_BRAKET_DEVICES = AMAZON_BRAKET_LOCAL_DEVICES + AMAZON_BRAKET_REMOTE_DEVICES
FALLBACK_AMAZON_BRAKET_LOCAL_BACKENDS = ["default", "braket_sv", "braket_dm", "braket_ahs"]


def _normalize_amazon_backend_name(name: str) -> str:
    name = name.removeprefix("_")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return name.lower()


def _build_amazon_braket_local_backends() -> list[str]:
    backends = sorted(entry.name for entry in entry_points(group="braket.simulators"))
    return backends or FALLBACK_AMAZON_BRAKET_LOCAL_BACKENDS


def _build_amazon_backend_key(provider_name: str, device_name: str | None = None) -> str:
    key = provider_name.lower()
    if device_name is not None:
        key = f"{key}_{_normalize_amazon_backend_name(device_name)}"
    return key


def _build_amazon_backend_types() -> dict[str, Any]:
    global Devices
    if Devices is None:
        try:
            from braket.devices import Devices as BraketDevices
        except ImportError as exc:
            raise ImportError(
                "Amazon Braket support requires optional dependencies. "
                'Install them with `pip install "qxmt[amazon-braket]"`.'
            ) from exc
        Devices = BraketDevices

    backend_types = {}

    for provider_name, provider in Devices.__dict__.items():
        if provider_name.startswith("_"):
            continue

        devices = getattr(provider, "__members__", None)
        if devices is None:
            continue

        default_device = None
        for device_name, device in devices.items():
            backend_types[_build_amazon_backend_key(provider_name, device_name)] = device
            if "_" not in device_name:
                default_device = device

        if default_device is not None:
            backend_types[_build_amazon_backend_key(provider_name)] = default_device

    for backend_name in ["sv1", "dm1", "tn1"]:
        amazon_backend = backend_types.get(_build_amazon_backend_key("Amazon", backend_name))
        if amazon_backend is not None:
            backend_types[backend_name] = amazon_backend

    return backend_types


class _LazyAmazonBackendTypes:
    def __init__(self) -> None:
        self._backend_types: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._backend_types is None:
            self._backend_types = _build_amazon_backend_types()
        return self._backend_types

    def __getitem__(self, key: str) -> Any:
        return self._load()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._load()

    def __repr__(self) -> str:
        return repr(self._load())


AMAZON_BRAKET_LOCAL_BACKENDS = _build_amazon_braket_local_backends()
AMAZON_BRAKET_REMOTE_SIMULATOR_BACKENDS = [
    "sv1",
    "dm1",
    "tn1",
    "amazon_sv1",
    "amazon_dm1",
    "amazon_tn1",
]
AMAZON_BRAKET_SIMULATOR_BACKENDS = AMAZON_BRAKET_LOCAL_BACKENDS + AMAZON_BRAKET_REMOTE_SIMULATOR_BACKENDS
AmazonBackendType = _LazyAmazonBackendTypes()

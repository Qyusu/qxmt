from typing import Any, Optional

from pydantic import BaseModel


class DeviceConfig(BaseModel):
    platform: str
    name: str
    n_qubits: int
    shots: Optional[int] = None


class FeatureMapConfig(BaseModel):
    name: str
    params: dict[str, Any]


class KernelConfig(BaseModel):
    name: str
    params: dict[str, Any]


class ModelConfig(BaseModel):
    name: str
    file_name: str
    params: dict[str, Any]
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None

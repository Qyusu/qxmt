from typing import Optional

from pydantic import BaseModel


class DeviceConfig(BaseModel):
    platform: str
    name: str
    n_qubits: int
    # shots: int


class ModelConfig(BaseModel):
    model_name: str
    model_file_name: str
    model_params: dict
    feature_map: Optional[str] = None
    map_params: dict
    kernel: Optional[str] = None

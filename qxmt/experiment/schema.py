from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from qxmt.datasets.schema import Dataset
from qxmt.models.qkernels import BaseMLModel
from qxmt.models.vqe import BaseVQE


class RemoteMachine(BaseModel):
    provider: str
    backend: str
    job_ids: list[str]


class RunTime(BaseModel):
    train_seconds: float
    validation_seconds: Optional[float]
    test_seconds: float


class VQERunTime(BaseModel):
    optimize_seconds: float
    circuit_depth: int
    n_parameters: int


class Evaluations(BaseModel):
    validation: Optional[dict[str, float]]
    test: dict[str, float]


class VQEEvaluations(BaseModel):
    optimized: dict[str, Optional[float]]


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: int
    desc: str
    remote_machine: Optional[RemoteMachine] = None
    commit_id: str
    config_file_name: Path
    execution_time: str
    runtime: RunTime | VQERunTime
    evaluations: Evaluations | VQEEvaluations


class RunArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    run_id: int
    dataset: Optional[Dataset]
    model: BaseMLModel | BaseVQE


class ExperimentDB(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    desc: str
    working_dirc: Path
    experiment_dirc: Path
    runs: list[RunRecord]

from pathlib import Path

from pydantic import BaseModel

from qxmt.datasets import Dataset
from qxmt.models import BaseMLModel


class RunRecord(BaseModel):
    run_id: int
    desc: str
    execution_time: str
    commit_id: str
    config_path: Path | str
    evaluation: dict[str, float]


class RunArtifact(BaseModel):
    run_id: int
    dataset: Dataset
    model: BaseMLModel

    class Config:
        arbitrary_types_allowed = True


class ExperimentDB(BaseModel):
    name: str
    desc: str
    working_dirc: Path
    experiment_dirc: Path
    runs: list[RunRecord]

from pathlib import Path

from pydantic import BaseModel


class RunRecord(BaseModel):
    run_id: int
    desc: str
    evaluation: dict[str, float]


class ExperimentDB(BaseModel):
    name: str
    desc: str
    experiment_dirc: Path
    runs: list[RunRecord]

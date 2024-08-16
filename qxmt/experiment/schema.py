from pathlib import Path

from pydantic import BaseModel


class RunRecord(BaseModel):
    run_id: int
    desc: str
    execution_time: str
    commit_id: str
    evaluation: dict[str, float]


class ExperimentDB(BaseModel):
    name: str
    desc: str
    working_dirc: Path
    experiment_dirc: Path
    runs: list[RunRecord]

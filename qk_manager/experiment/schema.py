from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    desc: str
    experiment_dirc: str


@dataclass(frozen=True)
class RunRecord:
    run_id: int


@dataclass(frozen=False)
class ExperimentDB:
    experiment_info: ExperimentRecord
    run_info: list[RunRecord]

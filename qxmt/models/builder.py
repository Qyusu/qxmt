from qxmt.configs import ExperimentConfig
from qxmt.constants import DEFAULT_N_JOBS
from qxmt.models.qkernels.base import BaseMLModel
from qxmt.models.qkernels.builder import KernelModelBuilder


class ModelBuilder:
    """Builder class for quantum machine learning models.
    This class is responsible for building quantum machine learning models.
    """

    def __init__(self, config: ExperimentConfig, n_jobs: int = DEFAULT_N_JOBS, show_progress: bool = True) -> None:
        self.config = config
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def build(self) -> BaseMLModel:
        model_type = self.config.global_settings.model_type
        match model_type:
            case "qkernel":
                return KernelModelBuilder(self.config, self.n_jobs, self.show_progress).build()
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from qxmt.constants import DEFAULT_EXP_CONFIG_FILE
from qxmt.exceptions import ReproductionError
from qxmt.experiment.schema import Evaluations, RunArtifact, RunRecord, VQEEvaluations
from qxmt.utils import get_commit_id, is_git_available

if TYPE_CHECKING:  # pragma: no cover
    # to avoid circular import
    from qxmt.experiment.experiment import Experiment

IS_GIT_AVAILABLE = is_git_available()


class Reproducer:
    """Handle reproduction of an existing experiment run.

    This class is responsible for reproducing and validating experiment runs.
    It is intentionally decoupled from the Experiment facade, but requires an
    Experiment instance to access database information and launch new runs.

    Attributes:
        _exp (Experiment): The experiment instance to use for reproduction.
        logger: Logger instance for logging operations.
    """

    def __init__(self, experiment: "Experiment") -> None:
        """Initialize the Reproducer.

        Args:
            experiment (Experiment): The experiment instance to use for reproduction.
                Note: Experiment is forward-declared via string to avoid circular import.
        """
        self._exp = experiment
        self.logger = experiment.logger

    def reproduce(
        self,
        run_id: int,
        *,
        check_commit_id: bool = False,
        tol: float = 1e-6,
    ) -> Tuple[RunArtifact, RunRecord]:
        """Reproduce a specific run and validate its evaluation metrics.

        This method reproduces a run with the given ID and compares its results
        with the original run to ensure reproducibility.

        Args:
            run_id (int): ID of the run to reproduce.
            check_commit_id (bool, optional): Whether to check if the current commit
                ID matches the original run's commit ID. Defaults to False.
            tol (float, optional): Tolerance for comparing evaluation metrics.
                Defaults to 1e-6.

        Returns:
            Tuple[RunArtifact, RunRecord]: A tuple containing the reproduced run's
                artifact and record.

        Raises:
            ReproductionError: If the run was executed in instance mode or if
                the config file is not found.
            ValueError: If the evaluation results differ beyond the specified
                tolerance or if the evaluation types are invalid.

        Examples:
            >>> reproducer = Reproducer(experiment)
            >>> artifact, record = reproducer.reproduce(run_id=1, check_commit_id=True)
        """
        self._exp._is_initialized()

        run_record = self._exp.get_run_record(self._exp.exp_db.runs, run_id)  # type: ignore

        # Optional commit‑id consistency check
        if check_commit_id:
            self._check_commit_id(run_id, run_record)

        # Runs executed via instance‑mode cannot be reproduced
        if run_record.config_file_name == Path(""):
            raise ReproductionError(
                f"run_id={run_id} does not have a config file path. This run executed from instance. "
                "Run from instance mode not supported for reproduction."
            )

        # Execute a new run with the saved config (results are *not* stored)
        config_path = (
            Path(f"{self._exp.experiment_dirc}/run_{run_id}/{DEFAULT_EXP_CONFIG_FILE}")
            if run_record.config_file_name != Path("")
            else None
        )
        reproduced_artifact, reproduced_result = self._exp.run(
            config_source=config_path,
            add_results=False,
        )

        # Validate evaluations
        if isinstance(run_record.evaluations, Evaluations) and isinstance(reproduced_result.evaluations, Evaluations):
            original_eval = run_record.evaluations.test
            new_eval = reproduced_result.evaluations.test
        elif isinstance(run_record.evaluations, VQEEvaluations) and isinstance(
            reproduced_result.evaluations, VQEEvaluations
        ):
            original_eval = run_record.evaluations.optimized
            new_eval = reproduced_result.evaluations.optimized
        else:
            raise ValueError(
                f"Invalid evaluation types: {type(run_record.evaluations)} vs {type(reproduced_result.evaluations)}"
            )

        self._validate_evaluation(original_eval, new_eval, tol)
        self.logger.info(f"Reproduce model successful. Evaluation results match run_id={run_id}.")
        return reproduced_artifact, reproduced_result

    def _check_commit_id(self, run_id: int, run_record: RunRecord) -> None:
        current_commit = get_commit_id() if IS_GIT_AVAILABLE else ""
        if current_commit != run_record.commit_id:
            self.logger.warning(
                f'Current commit_id="{current_commit}" differs from run_id={run_id} commit_id="{run_record.commit_id}".'
            )

    @staticmethod
    def _validate_evaluation(
        logging_evaluation: dict[str, float],
        reproduction_evaluation: dict[str, float],
        tol: float,
    ) -> None:
        """Compare two evaluation dictionaries within given tolerance.

        Args:
            logging_evaluation (dict[str, float]): Original evaluation results.
            reproduction_evaluation (dict[str, float]): Reproduced evaluation results.
            tol (float): Tolerance for comparing values.

        Raises:
            ValueError: If any metric is missing in the reproduction or if the
                values differ beyond the specified tolerance.
        """
        invalid: dict[str, str] = {}
        for key, value in logging_evaluation.items():
            rep_val = reproduction_evaluation.get(key)
            if rep_val is None:
                raise ValueError(f"Reproduction evaluation of {key} is not found.")
            if abs(value - rep_val) > tol:
                invalid[key] = f"{value} -> {rep_val}"

        if invalid:
            raise ValueError(
                f"Evaluation results differ between logging and reproduction (invalid metrics: {invalid})."
            )

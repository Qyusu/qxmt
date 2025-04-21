from typing import Any, Optional

from qxmt.evaluation import (
    ClassificationEvaluation,
    RegressionEvaluation,
    VQEEvaluation,
)

QKERNEL_MODEL_TYPE_NAME: str = "qkernel"
VQE_MODEL_TYPE_NAME: str = "vqe"


class EvaluationFactory:
    """Factory class that instantiates and executes appropriate evaluation.

    This class centralizes the mapping between (model_type, task_type) and concrete
    Evaluation implementation, decoupling Experiment/Executor classes from evaluation
    details and making it easy to add new metrics or model types.

    Class Attributes:
        QKERNEL_MODEL_TYPE_NAME (str): Constant representing the qkernel model type.
        VQE_MODEL_TYPE_NAME (str): Constant representing the VQE model type.
    """

    @staticmethod
    def evaluate(
        *,
        model_type: str,
        task_type: Optional[str],
        params: dict[str, Any],
        default_metrics_name: Optional[list[str]] = None,
        custom_metrics: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Perform evaluation and return result as plain dictionary.

        Args:
            model_type (str): Type of the model to evaluate. Must be either "qkernel" or "vqe".
            task_type (Optional[str]): Type of the task. For qkernel, must be "classification" or
                "regression". For vqe, must be None.
            params (dict[str, Any]): Arguments forwarded to the evaluation class. For supervised tasks,
                should include "actual" and "predicted". For VQE, should include "cost_history".
            default_metrics_name (Optional[list[str]]): List of default metric names to use.
            custom_metrics (Optional[list[dict[str, Any]]]): List of custom metrics to use.

        Returns:
            dict[str, Any]: Dictionary containing the evaluation results.

        Raises:
            ValueError: If the combination of model_type and task_type is invalid.

        Examples:
            >>> result = EvaluationFactory.evaluate(
            ...     model_type="qkernel",
            ...     task_type="classification",
            ...     params={"actual": [0, 1], "predicted": [0, 1]},
            ...     default_metrics_name=["accuracy"]
            ... )
        """
        if model_type == QKERNEL_MODEL_TYPE_NAME and task_type == "classification":
            evaluation = ClassificationEvaluation(
                params=params,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        elif model_type == QKERNEL_MODEL_TYPE_NAME and task_type == "regression":
            evaluation = RegressionEvaluation(
                params=params,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        elif model_type == VQE_MODEL_TYPE_NAME:
            evaluation = VQEEvaluation(
                params=params,
                default_metrics_name=default_metrics_name,
                custom_metrics=custom_metrics,
            )
        else:
            raise ValueError(f"Invalid model_type={model_type}, task_type={task_type}")

        evaluation.evaluate()
        return evaluation.to_dict()

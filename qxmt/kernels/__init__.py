from qxmt.kernels.base import BaseKernel
from qxmt.kernels.pennylane.base import PennyLaneBaseKernel
from qxmt.kernels.sampling import (
    generate_all_observable_states,
    sample_results_to_probs,
    validate_sampling_values,
)

__all__ = [
    "BaseKernel",
    "PennyLaneBaseKernel",
    "generate_all_observable_states",
    "sample_results_to_probs",
    "validate_sampling_values",
]


def __getattr__(name: str):
    if name == "QulacsBaseKernel":
        from qxmt.kernels.qulacs.base import QulacsBaseKernel

        return QulacsBaseKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

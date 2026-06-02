from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.feature_maps.pennylane.base import PennyLaneBaseFeatureMap

__all__ = ["BaseFeatureMap", "PennyLaneBaseFeatureMap"]


def __getattr__(name: str):
    if name == "QulacsBaseFeatureMap":
        from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap

        return QulacsBaseFeatureMap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

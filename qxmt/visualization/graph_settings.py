import numpy as np


def _create_colors(y: np.ndarray) -> dict[int, str]:
    return {class_value: f"C{i}" for i, class_value in enumerate(np.unique(y))}


def _create_class_labels(y: np.ndarray) -> dict[int, str]:
    return {class_value: str(class_value) for class_value in np.unique(y)}

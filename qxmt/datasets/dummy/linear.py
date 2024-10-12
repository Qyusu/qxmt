import numpy as np


def generate_linear_separable_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 3,
    noise: float = 0.1,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random linear separable data for multi-class classification.

    Args:
        n_samples (int, optional): sample size. Defaults to 100.
        n_features (int, optional): dimension of feature. Defaults to 2.
        n_classes (int, optional): number of classes. Defaults to 3.
        noise (float, optional): noise level. Defaults to 0.1.
        scale (float, optional): scale of data. Defaults to 1.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: generated data and label
    """
    X = scale * np.random.randn(n_samples, n_features)
    w = scale * np.random.randn(n_classes, n_features)

    # generate scores by multiplying feature matrix and weight matrix
    scores = X @ w.T + noise * np.random.randn(n_samples, n_classes)

    # assign the class with the highest score as the label
    y = np.argmax(scores, axis=1)

    return X, y

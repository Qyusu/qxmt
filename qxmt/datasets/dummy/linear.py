import numpy as np


def generate_linear_separable_data(
    n_samples: int = 100,
    n_features: int = 2,
    noise: float = 0.1,
    scale: float = 1.0,
    use_positive_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random linear separable data.

    Args:
        n_samples (int, optional): sample size. Defaults to 100.
        n_features (int, optional): dimension of feature. Defaults to 2.
        noise (float, optional): noise level. Defaults to 0.1.
        scale (float, optional): scale of data. Defaults to 1.0.
        use_positive_labels (bool, optional): use positive labels. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: generated data and label
    """
    X = scale * np.random.randn(n_samples, n_features)
    w = scale * np.random.randn(n_features)
    bias = scale * np.random.randn(n_samples) * noise
    y = np.sign(X @ w + bias)

    if use_positive_labels:
        y = np.where(y == -1, 0, 1)

    return X, y

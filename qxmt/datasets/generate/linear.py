from typing import Optional

import numpy as np


def generate_linear_separable_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 2,
    noise: float = 0.1,
    scale: float = 1.0,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random linear separable data for multi-class classification.

    Args:
        n_samples (int, optional): sample size. Defaults to 100.
        n_features (int, optional): dimension of feature. Defaults to 2.
        n_classes (int, optional): number of classes. Defaults to 2.
        noise (float, optional): noise level. Defaults to 0.1.
        scale (float, optional): scale of data. Defaults to 1.0.
        random_seed (int, optional): random seed. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: generated data and label
    """
    # set the random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    X = scale * np.random.randn(n_samples, n_features)
    w = scale * np.random.randn(n_classes, n_features)

    # generate scores by multiplying feature matrix and weight matrix
    scores = X @ w.T + noise * np.random.randn(n_samples, n_classes)

    # assign the class with the highest score as the label
    y = np.argmax(scores, axis=1)

    return X, y


def generate_linear_regression_data(
    n_samples: int = 100,
    n_features: int = 2,
    noise: float = 0.1,
    scale: float = 1.0,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random data for linear regression.

    Args:
        n_samples (int, optional): Number of samples. Defaults to 100.
        n_features (int, optional): Number of features. Defaults to 2.
        noise (float, optional): Noise level to add to the target. Defaults to 0.1.
        scale (float, optional): Scale of the data. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated features and target values
    """
    # set the random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # generate random feature data
    X = scale * np.random.randn(n_samples, n_features)

    # generate random weights for the linear relationship
    true_weights = scale * np.random.randn(n_features)

    # generate target variable with a linear relationship to the features
    y = X @ true_weights + noise * np.random.randn(n_samples)

    return X, y

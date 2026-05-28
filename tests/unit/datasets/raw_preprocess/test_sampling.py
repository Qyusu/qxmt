import numpy as np
import pytest

from qxmt.datasets.raw_preprocess.sampling import (
    sample_by_count,
    sample_n_per_class,
    sampling_by_each_class,
    sampling_by_num,
)


def test_sample_by_count_samples_n_samples() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_sampled, y_sampled = sample_by_count(X, y, n_samples=4, random_seed=42)

    assert X_sampled.shape == (4, 2)
    assert y_sampled.shape == (4,)
    assert len(np.unique(y_sampled)) == 4


def test_sample_by_count_raises_when_n_samples_exceeds_dataset_size() -> None:
    X = np.arange(8).reshape(4, 2)
    y = np.arange(4)

    with pytest.raises(ValueError):
        sample_by_count(X, y, n_samples=5, random_seed=42)


def test_sampling_by_num_warns_deprecation() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    with pytest.warns(FutureWarning, match="sampling_by_num is deprecated"):
        sampling_by_num(X, y, n_samples=4, random_seed=42)


def test_sample_n_per_class_samples_n_samples_per_class() -> None:
    X = np.arange(30).reshape(15, 2)
    y = np.array([0] * 5 + [1] * 5 + [2] * 5)

    X_sampled, y_sampled = sample_n_per_class(X, y, n_samples=2, labels=[0, 1, 2], random_seed=42)

    assert X_sampled.shape == (6, 2)
    assert y_sampled.shape == (6,)
    assert {label: int(np.sum(y_sampled == label)) for label in [0, 1, 2]} == {
        0: 2,
        1: 2,
        2: 2,
    }


def test_sample_n_per_class_raises_when_label_has_insufficient_samples() -> None:
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 0, 0, 1])

    with pytest.raises(ValueError, match="Label 1 has only 1 samples"):
        sample_n_per_class(X, y, n_samples=2, labels=[0, 1], random_seed=42)


def test_sampling_by_each_class_warns_deprecation() -> None:
    X = np.arange(30).reshape(15, 2)
    y = np.array([0] * 5 + [1] * 5 + [2] * 5)

    with pytest.warns(FutureWarning, match="sampling_by_each_class is deprecated"):
        sampling_by_each_class(X, y, n_samples=2, labels=[0, 1, 2], random_seed=42)

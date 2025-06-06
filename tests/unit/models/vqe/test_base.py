import numpy as np
import pytest

from qxmt.models.vqe.base import OptimizerPlatform


def test_vqe_init_defaults(build_vqe):
    vqe = build_vqe()
    assert vqe.max_steps == 100
    assert vqe.min_steps == 10
    assert vqe.tol == 1e-6
    assert vqe.verbose is True
    assert vqe.optimizer_settings is None
    assert vqe.init_params_config is None
    assert vqe.is_optimized() is False


def test_set_optimizer_platform_scipy(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    assert vqe.optimizer_platform == OptimizerPlatform.SCIPY


def test_set_optimizer_platform_pennylane(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "Adam"})
    assert vqe.optimizer_platform == OptimizerPlatform.PENNYLANE


def test_parse_init_params_zeros(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    params = vqe._parse_init_params({"type": "zeros"}, 3)
    assert np.allclose(params, np.zeros(3))


def test_parse_init_params_random(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    params = vqe._parse_init_params({"type": "random", "random_seed": 42}, 3)
    assert params.shape == (3,)
    assert np.allclose(params, np.array([0.77395605, 0.43887844, 0.85859792]))


def test_parse_init_params_custom(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    params = vqe._parse_init_params({"type": "custom", "values": [1, 2, 3]}, 3)
    assert np.allclose(params, np.array([1, 2, 3]))


def test_parse_init_params_custom_invalid(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    with pytest.raises(ValueError):
        vqe._parse_init_params({"type": "custom", "values": [1, 2]}, 3)


def test_is_optimized(build_vqe):
    vqe = build_vqe()
    assert not vqe.is_optimized()
    vqe.cost_history.append(-1.0)
    assert vqe.is_optimized()

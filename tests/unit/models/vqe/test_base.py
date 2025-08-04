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

    # check random seed
    params1 = vqe._parse_init_params({"type": "random", "random_seed": 42}, 3)
    params2 = vqe._parse_init_params({"type": "random", "random_seed": 42}, 3)
    assert np.allclose(params1, params2)

    # check defalt setting
    params = vqe._parse_init_params({"type": "random"}, 3)
    assert all(0.0 <= float(param) < 1.0 for param in params)

    # check max_value and min_value works
    max_value = 5.0
    min_value = 0.0
    params = vqe._parse_init_params(
        {"type": "random", "random_seed": 42, "max_value": max_value, "min_value": min_value}, 3
    )
    assert all(min_value <= float(param) < max_value for param in params)


def test_parse_init_params_custom(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    params = vqe._parse_init_params({"type": "custom", "values": [1, 2, 3]}, 3)
    assert np.allclose(params, np.array([1, 2, 3]))


def test_parse_init_params_custom_invalid(build_vqe):
    vqe = build_vqe(optimizer_settings={"name": "scipy.BFGS"})
    with pytest.raises(ValueError):
        vqe._parse_init_params({"type": "custom", "values": [1, 2]}, 3)


def test_is_params_updated(build_vqe):
    vqe = build_vqe()

    # only initial params
    assert not vqe.is_params_updated()

    # length of params_history is 2
    vqe.params_history.append(np.array([1, 2, 3]))
    assert not vqe.is_params_updated()

    # length of params_history is 3
    vqe.params_history.append(np.array([4, 5, 6]))
    assert vqe.is_params_updated()


def test_is_optimized(build_vqe):
    vqe = build_vqe()
    assert not vqe.is_optimized()
    vqe.cost_history.append(-1.0)
    assert vqe.is_optimized()

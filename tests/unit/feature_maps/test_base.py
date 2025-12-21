from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from qxmt.exceptions import InputShapeError, InvalidPlatformError
from qxmt.feature_maps.base import BaseFeatureMap, FeatureMapFromFunc
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class ConcreteFeatureMap(BaseFeatureMap):
    """Concrete implementation of BaseFeatureMap for testing."""

    def __init__(self, platform: str, n_qubits: int) -> None:
        super().__init__(platform, n_qubits)
        self.feature_map_called = False
        self.last_x: Optional[np.ndarray] = None

    def feature_map(self, x: np.ndarray) -> None:
        """Feature map implementation for testing."""
        self.feature_map_called = True
        self.last_x = x

    def draw(
        self,
        x: Optional[np.ndarray] = None,
        x_dim: Optional[int] = None,
        format: str = "default",
        logger: Any = LOGGER,
        **kwargs: Any,
    ) -> None:
        """Draw implementation for testing."""
        pass


class ConcreteFeatureMapFromFunc(FeatureMapFromFunc):
    """Concrete implementation of FeatureMapFromFunc with draw method for testing."""

    def __init__(self, platform: str, n_qubits: int, feature_map_func: Callable[[np.ndarray], None]) -> None:
        super().__init__(platform, n_qubits, feature_map_func)

    def draw(
        self,
        x: Optional[np.ndarray] = None,
        x_dim: Optional[int] = None,
        format: str = "default",
        logger: Any = LOGGER,
        **kwargs: Any,
    ) -> None:
        """Draw implementation for testing."""
        pass


@pytest.fixture(scope="function")
def concrete_feature_map() -> ConcreteFeatureMap:
    return ConcreteFeatureMap(platform="pennylane", n_qubits=2)


@pytest.fixture(scope="function")
def feature_map_from_func() -> ConcreteFeatureMapFromFunc:
    def test_func(x: np.ndarray) -> None:
        pass

    return ConcreteFeatureMapFromFunc(platform="pennylane", n_qubits=2, feature_map_func=test_func)


class TestBaseFeatureMap:
    def test__init__valid_platform(self, concrete_feature_map: ConcreteFeatureMap) -> None:
        """Test initialization with valid platform."""
        assert concrete_feature_map.platform == "pennylane"
        assert concrete_feature_map.n_qubits == 2

    def test__init__invalid_platform(self) -> None:
        """Test initialization with invalid platform raises error."""
        with pytest.raises(InvalidPlatformError) as exc_info:
            ConcreteFeatureMap(platform="invalid_platform", n_qubits=2)
        assert "invalid_platform" in str(exc_info.value)
        assert "not supported" in str(exc_info.value)

    def test__call__(self, concrete_feature_map: ConcreteFeatureMap) -> None:
        """Test __call__ method calls feature_map."""
        x = np.array([1.0, 2.0])
        concrete_feature_map(x)
        assert concrete_feature_map.feature_map_called is True
        assert concrete_feature_map.last_x is not None
        np.testing.assert_array_equal(concrete_feature_map.last_x, x)

    def test_check_input_dim_eq_nqubits_valid(self, concrete_feature_map: ConcreteFeatureMap) -> None:
        """Test check_input_dim_eq_nqubits with valid input dimension."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2), last dim = 2
        # Should not raise error
        concrete_feature_map.check_input_dim_eq_nqubits(x)

    def test_check_input_dim_eq_nqubits_invalid(self, concrete_feature_map: ConcreteFeatureMap) -> None:
        """Test check_input_dim_eq_nqubits with invalid input dimension."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape: (2, 3), last dim = 3
        with pytest.raises(InputShapeError) as exc_info:
            concrete_feature_map.check_input_dim_eq_nqubits(x)
        assert "Input data shape does not match the number of qubits" in str(exc_info.value)

    def test_check_input_dim_eq_nqubits_custom_idx(self, concrete_feature_map: ConcreteFeatureMap) -> None:
        """Test check_input_dim_eq_nqubits with custom index."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2)
        # Check first dimension (idx=0)
        concrete_feature_map.n_qubits = 2
        concrete_feature_map.check_input_dim_eq_nqubits(x, idx=0)

        # Check with invalid first dimension
        x_invalid = np.array([[1.0, 2.0]])  # shape: (1, 2)
        concrete_feature_map.n_qubits = 2
        with pytest.raises(InputShapeError):
            concrete_feature_map.check_input_dim_eq_nqubits(x_invalid, idx=0)

    def test_feature_map_abstract(self) -> None:
        """Test that BaseFeatureMap cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFeatureMap(platform="pennylane", n_qubits=2)  # type: ignore


class TestFeatureMapFromFunc:
    def test__init__(self, feature_map_from_func: ConcreteFeatureMapFromFunc) -> None:
        """Test FeatureMapFromFunc initialization."""
        assert feature_map_from_func.platform == "pennylane"
        assert feature_map_from_func.n_qubits == 2
        assert feature_map_from_func.feature_map_func is not None

    def test_feature_map_calls_func(self, feature_map_from_func: ConcreteFeatureMapFromFunc) -> None:
        """Test that feature_map calls the wrapped function."""
        mock_func = MagicMock()
        feature_map_from_func.feature_map_func = mock_func
        x = np.array([1.0, 2.0])
        feature_map_from_func.feature_map(x)
        mock_func.assert_called_once_with(x)

    def test__call__calls_feature_map(self, feature_map_from_func: ConcreteFeatureMapFromFunc) -> None:
        """Test that __call__ calls feature_map which calls the wrapped function."""
        mock_func = MagicMock()
        feature_map_from_func.feature_map_func = mock_func
        x = np.array([1.0, 2.0])
        feature_map_from_func(x)
        mock_func.assert_called_once_with(x)

    def test_invalid_platform(self) -> None:
        """Test FeatureMapFromFunc with invalid platform raises error."""

        def test_func(x: np.ndarray) -> None:
            pass

        with pytest.raises(InvalidPlatformError):
            ConcreteFeatureMapFromFunc(platform="invalid_platform", n_qubits=2, feature_map_func=test_func)

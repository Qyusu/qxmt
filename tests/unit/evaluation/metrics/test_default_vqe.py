from unittest.mock import MagicMock

import pytest

from qxmt.evaluation.metrics.default_vqe import FCIEnergy, FinalCost, HFEnergy


class TestFinalCost:
    @pytest.mark.parametrize(
        ["cost_history", "expected"],
        [
            pytest.param([1.0, 2.0, 3.0], 3.0, id="decreasing"),
            pytest.param([1.0, 2.0, 3.0, 4.0], 4.0, id="increasing"),
            pytest.param([1.0, 1.0, 1.0, 1.0], 1.0, id="constant"),
            pytest.param([1.0, 2.0, 3.0, 2.0, 1.0], 1.0, id="oscillating"),
            pytest.param([1.0, -1.5, -3.0], -3.0, id="negative_values"),
        ],
    )
    def test_evaluate(self, cost_history: list[float], expected: float) -> None:
        final_cost = FinalCost()
        result = final_cost.evaluate(cost_history)
        assert result == expected


class TestHFEnergy:
    @pytest.mark.parametrize(
        ["hf_energy", "expected"],
        [
            pytest.param(-1.0, -1.0, id="negative_energy"),
            pytest.param(0.0, 0.0, id="zero_energy"),
            pytest.param(1.0, 1.0, id="positive_energy"),
        ],
    )
    def test_evaluate(self, hf_energy: float, expected: float) -> None:
        mock_hamiltonian = MagicMock()
        mock_hamiltonian.get_hf_energy.return_value = hf_energy

        hf_energy_metric = HFEnergy()
        result = hf_energy_metric.evaluate(mock_hamiltonian)

        assert result == expected
        mock_hamiltonian.get_hf_energy.assert_called_once()


class TestFCIEnergy:
    @pytest.mark.parametrize(
        ["fci_energy", "expected"],
        [
            pytest.param(-1.0, -1.0, id="negative_energy"),
            pytest.param(0.0, 0.0, id="zero_energy"),
            pytest.param(1.0, 1.0, id="positive_energy"),
            pytest.param(None, None, id="none_energy"),
        ],
    )
    def test_evaluate(self, fci_energy: float | None, expected: float | None) -> None:
        mock_hamiltonian = MagicMock()
        mock_hamiltonian.get_fci_energy.return_value = fci_energy

        fci_energy_metric = FCIEnergy()
        result = fci_energy_metric.evaluate(mock_hamiltonian)

        assert result == expected
        mock_hamiltonian.get_fci_energy.assert_called_once()

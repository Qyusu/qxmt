"""Energy data structures for quantum chemistry calculations."""

from typing import Optional

from pydantic import BaseModel, Field


class ReferenceEnergies(BaseModel):
    """Data structure for storing reference energies from quantum chemistry calculations.

    This class encapsulates all the reference energies computed during quantum chemistry
    calculations, providing a clean interface for accessing and managing energy values.

    Attributes:
        hf_energy: Hartree-Fock energy (mean-field ground state).
        casci_energy: Complete Active Space Configuration Interaction energy.
        casscf_energy: Complete Active Space Self-Consistent Field energy.
        fci_energy: Full Configuration Interaction energy (exact ground state).
    """

    hf_energy: Optional[float] = Field(
        default=None, description="Hartree-Fock energy representing the mean-field approximation"
    )
    casci_energy: Optional[float] = Field(
        default=None, description="CASCI energy from configuration interaction within active space"
    )
    casscf_energy: Optional[float] = Field(
        default=None, description="CASSCF energy with orbital optimization in active space"
    )
    fci_energy: Optional[float] = Field(
        default=None, description="Full CI energy representing the exact solution within basis set"
    )

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "hf_energy": self.hf_energy,
            "casci_energy": self.casci_energy,
            "casscf_energy": self.casscf_energy,
            "fci_energy": self.fci_energy,
        }

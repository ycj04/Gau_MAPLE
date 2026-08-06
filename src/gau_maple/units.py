"""Unit conversions at the Gaussian External ↔ MAPLE boundary."""

from __future__ import annotations

import numpy as np

from .errors import UnitConversionError

# CODATA value used by ASE and common electronic-structure interfaces.
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


def _finite(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise UnitConversionError(f"{name} contains NaN or infinity.")
    return array


def positions_bohr_to_angstrom(positions_bohr) -> np.ndarray:
    return _finite(positions_bohr, name="positions_bohr") * BOHR_TO_ANGSTROM


def positions_angstrom_to_bohr(positions_angstrom) -> np.ndarray:
    return _finite(positions_angstrom, name="positions_angstrom") * ANGSTROM_TO_BOHR


def maple_forces_to_gaussian_gradient(forces_hartree_per_angstrom) -> np.ndarray:
    """Convert MAPLE force to Gaussian gradient.

    MAPLE force is -dE/dx in Hartree/Angstrom.
    Gaussian External expects dE/dq in Hartree/Bohr.
    """
    forces = _finite(
        forces_hartree_per_angstrom,
        name="forces_hartree_per_angstrom",
    )
    return -forces * BOHR_TO_ANGSTROM


def gaussian_gradient_to_maple_forces(gradient_hartree_per_bohr) -> np.ndarray:
    gradient = _finite(
        gradient_hartree_per_bohr,
        name="gradient_hartree_per_bohr",
    )
    return -gradient * ANGSTROM_TO_BOHR


def maple_hessian_to_gaussian(hessian_hartree_per_angstrom2) -> np.ndarray:
    """Convert Hartree/Angstrom^2 to Hartree/Bohr^2."""
    hessian = _finite(
        hessian_hartree_per_angstrom2,
        name="hessian_hartree_per_angstrom2",
    )
    return hessian * (BOHR_TO_ANGSTROM**2)


def gaussian_hessian_to_maple(hessian_hartree_per_bohr2) -> np.ndarray:
    """Convert Hartree/Bohr^2 to Hartree/Angstrom^2."""
    hessian = _finite(
        hessian_hartree_per_bohr2,
        name="hessian_hartree_per_bohr2",
    )
    return hessian * (ANGSTROM_TO_BOHR**2)

"""Typed data models for the Gaussian External protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import ExternalFormatError


def _finite_array(
    value: Iterable[float] | np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ExternalFormatError(
            f"{name} must have shape {shape}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ExternalFormatError(f"{name} contains NaN or infinity.")
    return array


@dataclass(frozen=True, slots=True)
class ExternalRequest:
    """A parsed Gaussian External input request.

    Coordinates are stored exactly as Gaussian supplies them: Bohr.
    """

    atomic_numbers: np.ndarray
    positions_bohr: np.ndarray
    derivative_order: int
    charge: int
    multiplicity: int
    mm_charges: np.ndarray
    extra_header_fields: tuple[int, ...] = field(default_factory=tuple)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        atomic_numbers = np.asarray(self.atomic_numbers, dtype=np.int64)
        if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
            raise ExternalFormatError(
                "atomic_numbers must be a non-empty one-dimensional array."
            )
        if np.any(atomic_numbers <= 0):
            raise ExternalFormatError("All atomic numbers must be positive integers.")

        natoms = int(atomic_numbers.size)
        positions = _finite_array(
            self.positions_bohr,
            name="positions_bohr",
            shape=(natoms, 3),
        )
        mm_charges = _finite_array(
            self.mm_charges,
            name="mm_charges",
            shape=(natoms,),
        )

        if self.derivative_order not in (0, 1, 2):
            raise ExternalFormatError(
                "derivative_order must be 0 (energy), 1 (gradient), or 2 (Hessian)."
            )
        if self.multiplicity < 1:
            raise ExternalFormatError("multiplicity must be at least 1.")

        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "positions_bohr", positions)
        object.__setattr__(self, "mm_charges", mm_charges)
        object.__setattr__(
            self,
            "extra_header_fields",
            tuple(int(x) for x in self.extra_header_fields),
        )

    @property
    def natoms(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def ndof(self) -> int:
        return 3 * self.natoms


@dataclass(slots=True)
class ExternalResult:
    """Results in the atomic units required by Gaussian External.

    Required units:
      * energy_hartree: Hartree
      * gradient_hartree_per_bohr: Hartree / Bohr
      * hessian_hartree_per_bohr2: Hartree / Bohr^2
      * dipole_au, polarizability_au, dipole_derivatives_au: atomic units
    """

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None
    dipole_au: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    polarizability_au: np.ndarray | None = None
    dipole_derivatives_au: np.ndarray | None = None

    def validated_for(self, request: ExternalRequest) -> "ExternalResult":
        energy = float(self.energy_hartree)
        if not np.isfinite(energy):
            raise ExternalFormatError("energy_hartree must be finite.")
        self.energy_hartree = energy

        self.dipole_au = _finite_array(
            self.dipole_au,
            name="dipole_au",
            shape=(3,),
        )

        if request.derivative_order >= 1:
            if self.gradient_hartree_per_bohr is None:
                raise ExternalFormatError(
                    "A derivative-order 1 or 2 request requires a gradient."
                )
            self.gradient_hartree_per_bohr = _finite_array(
                self.gradient_hartree_per_bohr,
                name="gradient_hartree_per_bohr",
                shape=(request.natoms, 3),
            )
        elif self.gradient_hartree_per_bohr is not None:
            self.gradient_hartree_per_bohr = _finite_array(
                self.gradient_hartree_per_bohr,
                name="gradient_hartree_per_bohr",
                shape=(request.natoms, 3),
            )

        if request.derivative_order == 2:
            if self.hessian_hartree_per_bohr2 is None:
                raise ExternalFormatError(
                    "A derivative-order 2 request requires a Hessian."
                )
            hessian = _finite_array(
                self.hessian_hartree_per_bohr2,
                name="hessian_hartree_per_bohr2",
                shape=(request.ndof, request.ndof),
            )
            if not np.allclose(hessian, hessian.T, rtol=1.0e-8, atol=1.0e-10):
                max_asymmetry = float(np.max(np.abs(hessian - hessian.T)))
                raise ExternalFormatError(
                    "hessian_hartree_per_bohr2 is not symmetric; "
                    f"maximum |H-H.T|={max_asymmetry:.3e}."
                )
            self.hessian_hartree_per_bohr2 = hessian

            if self.polarizability_au is None:
                self.polarizability_au = np.zeros(6, dtype=np.float64)
            else:
                self.polarizability_au = _finite_array(
                    self.polarizability_au,
                    name="polarizability_au",
                    shape=(6,),
                )

            if self.dipole_derivatives_au is None:
                self.dipole_derivatives_au = np.zeros(
                    9 * request.natoms,
                    dtype=np.float64,
                )
            else:
                ddip = np.asarray(self.dipole_derivatives_au, dtype=np.float64)
                if ddip.shape == (request.ndof, 3):
                    ddip = ddip.reshape(-1)
                if ddip.shape != (9 * request.natoms,):
                    raise ExternalFormatError(
                        "dipole_derivatives_au must have shape "
                        f"({9 * request.natoms},) or ({request.ndof}, 3), "
                        f"got {ddip.shape}."
                    )
                if not np.all(np.isfinite(ddip)):
                    raise ExternalFormatError(
                        "dipole_derivatives_au contains NaN or infinity."
                    )
                self.dipole_derivatives_au = ddip

        return self

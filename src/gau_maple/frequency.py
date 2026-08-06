"""Independent Hessian and vibrational-frequency diagnostics for Gau_MAPLE.

The production Gaussian path only needs to emit a Cartesian Hessian.  This module
adds a second, independent validation layer that can:

* finite-difference Gaussian-unit gradients returned by a running server;
* compare the finite-difference Hessian with the backend Hessian;
* project translations/rotations from a mass-weighted Hessian; and
* report signed harmonic wavenumbers in cm^-1.

All Hessians accepted by this module use the Gaussian External convention:
Hartree / Bohr^2.  Coordinates use Bohr and masses use unified atomic mass
units (u, historically called amu).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np

from .errors import BackendExecutionError
from .models import ExternalRequest, ExternalResult

AMU_TO_ELECTRON_MASS = 1822.888486209
ATOMIC_TIME_SECONDS = 2.4188843265864e-17
SPEED_OF_LIGHT_CM_S = 2.99792458e10
AU_ANGULAR_FREQUENCY_TO_WAVENUMBER = 1.0 / (
    2.0 * np.pi * ATOMIC_TIME_SECONDS * SPEED_OF_LIGHT_CM_S
)


@dataclass(frozen=True, slots=True)
class FiniteDifferenceHessian:
    """Central finite-difference Hessian and its raw symmetry diagnostic."""

    hessian_hartree_per_bohr2: np.ndarray
    raw_max_asymmetry: float
    step_bohr: float
    gradient_evaluations: int


@dataclass(frozen=True, slots=True)
class HessianComparison:
    """Absolute comparison metrics between two Cartesian Hessians."""

    max_abs_error: float
    rms_error: float
    mean_abs_error: float
    reference_max_abs: float
    normalized_rms_error: float


@dataclass(frozen=True, slots=True)
class FrequencyAnalysis:
    """Projected harmonic analysis from a Cartesian Hessian."""

    frequencies_cm1: np.ndarray
    all_projected_eigenvalues_au: np.ndarray
    vibrational_eigenvalues_au: np.ndarray
    external_mode_rank: int
    is_linear: bool
    translation_residual: float

    @property
    def imaginary_count(self) -> int:
        return int(np.count_nonzero(self.frequencies_cm1 < 0.0))


def _finite_matrix(value: np.ndarray, *, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise BackendExecutionError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise BackendExecutionError(f"{name} contains NaN or infinity.")
    return array


def finite_difference_hessian(
    request: ExternalRequest,
    evaluator: Callable[[ExternalRequest], ExternalResult],
    *,
    step_bohr: float = 1.0e-3,
) -> FiniteDifferenceHessian:
    """Differentiate Gaussian-unit gradients with central finite differences.

    ``evaluator`` must return a gradient in Hartree/Bohr for derivative-order 1
    requests.  Because both coordinates and gradients are already in Gaussian
    atomic units, no additional unit conversion appears in this independent
    check.
    """

    step = float(step_bohr)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"step_bohr must be a positive finite number, got {step_bohr!r}.")

    ndof = request.ndof
    base_positions = np.asarray(request.positions_bohr, dtype=np.float64)
    hessian = np.empty((ndof, ndof), dtype=np.float64)

    for column in range(ndof):
        atom, axis = divmod(column, 3)
        plus_positions = base_positions.copy()
        minus_positions = base_positions.copy()
        plus_positions[atom, axis] += step
        minus_positions[atom, axis] -= step

        plus_request = replace(
            request,
            positions_bohr=plus_positions,
            derivative_order=1,
            source_path=None,
        )
        minus_request = replace(
            request,
            positions_bohr=minus_positions,
            derivative_order=1,
            source_path=None,
        )

        plus_result = evaluator(plus_request).validated_for(plus_request)
        minus_result = evaluator(minus_request).validated_for(minus_request)
        assert plus_result.gradient_hartree_per_bohr is not None
        assert minus_result.gradient_hartree_per_bohr is not None
        plus_gradient = plus_result.gradient_hartree_per_bohr.reshape(-1)
        minus_gradient = minus_result.gradient_hartree_per_bohr.reshape(-1)
        hessian[:, column] = (plus_gradient - minus_gradient) / (2.0 * step)

    raw_asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    symmetric = 0.5 * (hessian + hessian.T)
    return FiniteDifferenceHessian(
        hessian_hartree_per_bohr2=symmetric,
        raw_max_asymmetry=raw_asymmetry,
        step_bohr=step,
        gradient_evaluations=2 * ndof,
    )


def compare_hessians(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> HessianComparison:
    """Return stable absolute and scale-normalized Hessian errors."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape or candidate_array.ndim != 2:
        raise ValueError(
            "candidate and reference must be two-dimensional arrays with the same shape; "
            f"got {candidate_array.shape} and {reference_array.shape}."
        )
    if candidate_array.shape[0] != candidate_array.shape[1]:
        raise ValueError("Hessians must be square matrices.")
    if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(reference_array)):
        raise ValueError("Hessians must contain only finite values.")

    difference = candidate_array - reference_array
    max_abs = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(np.square(difference))))
    mean_abs = float(np.mean(np.abs(difference)))
    reference_scale = float(np.max(np.abs(reference_array)))
    normalized = rms / max(reference_scale, np.finfo(np.float64).tiny)
    return HessianComparison(
        max_abs_error=max_abs,
        rms_error=rms,
        mean_abs_error=mean_abs,
        reference_max_abs=reference_scale,
        normalized_rms_error=float(normalized),
    )


def atomic_masses_amu(atomic_numbers: Iterable[int]) -> np.ndarray:
    """Look up standard ASE masses without importing ASE at package import time."""

    try:
        from ase.data import atomic_masses
    except Exception as exc:  # pragma: no cover - Gau_MAPLE declares ASE dependency
        raise BackendExecutionError("ASE atomic mass data are unavailable.") from exc

    numbers = np.asarray(tuple(int(value) for value in atomic_numbers), dtype=np.int64)
    if numbers.ndim != 1 or numbers.size == 0:
        raise ValueError("atomic_numbers must be a non-empty one-dimensional sequence.")
    if np.any(numbers <= 0) or np.any(numbers >= len(atomic_masses)):
        raise ValueError("atomic_numbers contains an unsupported atomic number.")
    masses = np.asarray(atomic_masses[numbers], dtype=np.float64)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("ASE returned invalid atomic masses.")
    return masses


def _orthonormal_columns(matrix: np.ndarray, *, relative_tolerance: float = 1.0e-10) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    rank = int(np.count_nonzero(singular_values > singular_values[0] * relative_tolerance))
    return u[:, :rank]


def external_mode_basis(
    positions_bohr: np.ndarray,
    masses_amu: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Construct orthonormal mass-weighted translation/rotation vectors.

    The returned basis has rank 3 for a monatomic system, 5 for a linear
    polyatomic system, and 6 for a nonlinear polyatomic system.
    """

    positions = np.asarray(positions_bohr, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    natoms = masses.size
    if positions.shape != (natoms, 3):
        raise ValueError(f"positions_bohr must have shape ({natoms}, 3), got {positions.shape}.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_bohr contains NaN or infinity.")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_amu must contain positive finite values.")

    center_of_mass = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    centered = positions - center_of_mass
    sqrt_mass = np.sqrt(masses)

    vectors: list[np.ndarray] = []
    for axis in np.eye(3):
        vectors.append((sqrt_mass[:, None] * axis[None, :]).reshape(-1))

    if natoms > 1:
        for axis in np.eye(3):
            displacement = np.cross(axis[None, :], centered)
            vectors.append((sqrt_mass[:, None] * displacement).reshape(-1))

    raw = np.column_stack(vectors)
    basis = _orthonormal_columns(raw)
    is_linear = natoms > 1 and basis.shape[1] == 5
    return basis, is_linear


def translation_residual(
    hessian_hartree_per_bohr2: np.ndarray,
    natoms: int,
) -> float:
    """Return max ||H t||/||t|| over the three Cartesian translations."""

    hessian = _finite_matrix(
        hessian_hartree_per_bohr2,
        shape=(3 * natoms, 3 * natoms),
        name="Hessian",
    )
    residuals: list[float] = []
    for axis in range(3):
        vector = np.zeros((natoms, 3), dtype=np.float64)
        vector[:, axis] = 1.0
        flat = vector.reshape(-1)
        residuals.append(float(np.linalg.norm(hessian @ flat) / np.linalg.norm(flat)))
    return max(residuals)


def harmonic_frequency_analysis(
    hessian_hartree_per_bohr2: np.ndarray,
    positions_bohr: np.ndarray,
    masses_amu: np.ndarray,
) -> FrequencyAnalysis:
    """Project external modes and compute signed harmonic wavenumbers.

    Negative wavenumbers represent imaginary modes.  Intensities are not
    computed: Gau_MAPLE currently writes zero dipole derivatives and zero
    polarizability placeholders to Gaussian.
    """

    masses = np.asarray(masses_amu, dtype=np.float64)
    natoms = int(masses.size)
    hessian = _finite_matrix(
        hessian_hartree_per_bohr2,
        shape=(3 * natoms, 3 * natoms),
        name="Hessian",
    )
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_amu must contain positive finite values.")

    basis, is_linear = external_mode_basis(positions_bohr, masses)
    mass_electron = np.repeat(masses * AMU_TO_ELECTRON_MASS, 3)
    inverse_sqrt_mass = 1.0 / np.sqrt(mass_electron)
    mass_weighted = hessian * inverse_sqrt_mass[:, None] * inverse_sqrt_mass[None, :]

    identity = np.eye(3 * natoms, dtype=np.float64)
    projector = identity - basis @ basis.T
    projected = projector @ mass_weighted @ projector
    projected = 0.5 * (projected + projected.T)
    eigenvalues = np.linalg.eigvalsh(projected)

    external_rank = basis.shape[1]
    if external_rank:
        remove = np.argsort(np.abs(eigenvalues))[:external_rank]
        keep_mask = np.ones(eigenvalues.size, dtype=bool)
        keep_mask[remove] = False
        vibrational = eigenvalues[keep_mask]
    else:
        vibrational = eigenvalues
    vibrational = np.sort(vibrational)
    frequencies = np.sign(vibrational) * np.sqrt(np.abs(vibrational))
    frequencies *= AU_ANGULAR_FREQUENCY_TO_WAVENUMBER

    return FrequencyAnalysis(
        frequencies_cm1=frequencies,
        all_projected_eigenvalues_au=eigenvalues,
        vibrational_eigenvalues_au=vibrational,
        external_mode_rank=external_rank,
        is_linear=is_linear,
        translation_residual=translation_residual(hessian, natoms),
    )

from __future__ import annotations

import numpy as np
import pytest

from gau_maple.frequency import (
    AMU_TO_ELECTRON_MASS,
    AU_ANGULAR_FREQUENCY_TO_WAVENUMBER,
    compare_hessians,
    external_mode_basis,
    finite_difference_hessian,
    harmonic_frequency_analysis,
)
from gau_maple.models import ExternalRequest, ExternalResult


def make_request(positions: np.ndarray, derivative_order: int = 2) -> ExternalRequest:
    natoms = positions.shape[0]
    return ExternalRequest(
        atomic_numbers=np.ones(natoms, dtype=int),
        positions_bohr=np.asarray(positions, dtype=float),
        derivative_order=derivative_order,
        charge=0,
        multiplicity=1,
        mm_charges=np.zeros(natoms),
    )


def test_finite_difference_hessian_recovers_quadratic_matrix() -> None:
    positions = np.array([[0.1, -0.2, 0.3]])
    request = make_request(positions)
    hessian = np.array(
        [
            [2.0, 0.2, -0.1],
            [0.2, 3.0, 0.4],
            [-0.1, 0.4, 4.0],
        ]
    )
    calls: list[ExternalRequest] = []

    def evaluator(req: ExternalRequest) -> ExternalResult:
        calls.append(req)
        flat = req.positions_bohr.reshape(-1)
        gradient = (hessian @ flat).reshape(req.natoms, 3)
        return ExternalResult(
            energy_hartree=0.5 * float(flat @ hessian @ flat),
            gradient_hartree_per_bohr=gradient,
        )

    result = finite_difference_hessian(request, evaluator, step_bohr=1.0e-4)
    np.testing.assert_allclose(result.hessian_hartree_per_bohr2, hessian, atol=1.0e-11)
    assert result.gradient_evaluations == 6
    assert len(calls) == 6
    assert result.raw_max_asymmetry < 1.0e-10


def test_finite_difference_rejects_nonpositive_step() -> None:
    request = make_request(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="positive finite"):
        finite_difference_hessian(request, lambda _: None, step_bohr=0.0)  # type: ignore[arg-type]


def test_compare_hessians_metrics() -> None:
    reference = np.eye(2)
    candidate = reference + np.array([[0.1, -0.2], [0.0, 0.3]])
    report = compare_hessians(candidate, reference)
    assert report.max_abs_error == pytest.approx(0.3)
    assert report.mean_abs_error == pytest.approx(0.15)
    assert report.rms_error == pytest.approx(np.sqrt((0.01 + 0.04 + 0.0 + 0.09) / 4.0))
    assert report.reference_max_abs == pytest.approx(1.0)
    assert report.normalized_rms_error == pytest.approx(report.rms_error)


def test_external_mode_rank_nonlinear_and_linear() -> None:
    masses = np.array([16.0, 1.0, 1.0])
    water = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.3, 1.3, 0.0]])
    basis, linear = external_mode_basis(water, masses)
    assert basis.shape == (9, 6)
    assert not linear
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1.0e-12)

    diatomic = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    basis, linear = external_mode_basis(diatomic, np.array([1.0, 1.0]))
    assert basis.shape == (6, 5)
    assert linear


def test_diatomic_harmonic_frequency() -> None:
    # One stretch coordinate with force constant k in Hartree/Bohr^2.
    k = 0.5
    hessian = np.zeros((6, 6))
    hessian[0, 0] = k
    hessian[0, 3] = -k
    hessian[3, 0] = -k
    hessian[3, 3] = k
    positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])

    analysis = harmonic_frequency_analysis(hessian, positions, masses)
    assert analysis.external_mode_rank == 5
    assert analysis.is_linear
    assert analysis.frequencies_cm1.shape == (1,)

    expected_eigenvalue = 2.0 * k / AMU_TO_ELECTRON_MASS
    expected_frequency = np.sqrt(expected_eigenvalue) * AU_ANGULAR_FREQUENCY_TO_WAVENUMBER
    assert analysis.frequencies_cm1[0] == pytest.approx(expected_frequency, rel=1.0e-10)
    assert analysis.imaginary_count == 0
    assert analysis.translation_residual == pytest.approx(0.0, abs=1.0e-14)


def test_negative_curvature_is_reported_as_imaginary() -> None:
    k = -0.5
    hessian = np.zeros((6, 6))
    hessian[0, 0] = k
    hessian[0, 3] = -k
    hessian[3, 0] = -k
    hessian[3, 3] = k
    analysis = harmonic_frequency_analysis(
        hessian,
        np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
    )
    assert analysis.frequencies_cm1[0] < 0.0
    assert analysis.imaginary_count == 1

import numpy as np

from gau_maple.units import (
    BOHR_TO_ANGSTROM,
    gaussian_gradient_to_maple_forces,
    gaussian_hessian_to_maple,
    maple_forces_to_gaussian_gradient,
    maple_hessian_to_gaussian,
    positions_angstrom_to_bohr,
    positions_bohr_to_angstrom,
)


def test_position_round_trip():
    positions = np.array([[0.0, 1.0, -2.0], [3.5, 0.2, 8.0]])
    assert np.allclose(
        positions_angstrom_to_bohr(positions_bohr_to_angstrom(positions)),
        positions,
    )


def test_force_gradient_sign_and_scale():
    forces = np.array([[1.0, -2.0, 0.5]])
    gradient = maple_forces_to_gaussian_gradient(forces)
    assert np.allclose(gradient, -forces * BOHR_TO_ANGSTROM)
    assert np.allclose(gaussian_gradient_to_maple_forces(gradient), forces)


def test_hessian_round_trip():
    hessian = np.arange(16.0).reshape(4, 4)
    converted = maple_hessian_to_gaussian(hessian)
    assert np.allclose(converted, hessian * BOHR_TO_ANGSTROM**2)
    assert np.allclose(gaussian_hessian_to_maple(converted), hessian)

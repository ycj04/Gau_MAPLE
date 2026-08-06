from pathlib import Path

import numpy as np
import pytest

from gau_maple.errors import ExternalFormatError
from gau_maple.gaussian_io import (
    parse_external_input,
    parse_external_output,
    write_external_output,
)
from gau_maple.models import ExternalResult

FIXTURES = Path(__file__).parent / "fixtures"


def _symmetric_hessian(ndof: int) -> np.ndarray:
    raw = np.arange(1, ndof * ndof + 1, dtype=float).reshape(ndof, ndof)
    return 0.5 * (raw + raw.T) / 1000.0


def test_parse_ein():
    request = parse_external_input(FIXTURES / "water_deriv2.EIn")
    assert request.natoms == 3
    assert request.derivative_order == 2
    assert request.charge == 0
    assert request.multiplicity == 1
    assert request.atomic_numbers.tolist() == [8, 1, 1]
    assert request.positions_bohr.shape == (3, 3)
    assert np.allclose(request.mm_charges, 0.0)




def test_accepts_gaussian16_trailing_metadata(tmp_path):
    source = tmp_path / "g16_trailing.EIn"
    source.write_text(
        "         3         0         0         1\n"
        "         8      0.000000000000      0.000000000000      0.000000000000      0.000000000000\n"
        "         1      1.430825000000      0.000000000000      1.107862000000      0.000000000000\n"
        "         1     -1.430825000000      0.000000000000      1.107862000000      0.000000000000\n"
        "Gaussian-16 implementation metadata\n"
        "0 0 0\n",
        encoding="ascii",
    )
    request = parse_external_input(source)
    assert request.natoms == 3
    assert request.atomic_numbers.tolist() == [8, 1, 1]
    assert request.derivative_order == 0


def test_trailing_metadata_does_not_relax_required_atom_validation(tmp_path):
    source = tmp_path / "truncated.EIn"
    source.write_text(
        "         3         0         0         1\n"
        "         8      0.000000000000      0.000000000000      0.000000000000      0.000000000000\n"
        "         1      1.430825000000      0.000000000000      1.107862000000      0.000000000000\n",
        encoding="ascii",
    )
    with pytest.raises(ExternalFormatError, match="ended after 2 atom records"):
        parse_external_input(source)


def test_deriv2_round_trip(tmp_path):
    request = parse_external_input(FIXTURES / "water_deriv2.EIn")
    gradient = np.arange(request.ndof, dtype=float).reshape(request.natoms, 3) / 100.0
    hessian = _symmetric_hessian(request.ndof)
    result = ExternalResult(
        energy_hartree=-76.123456789,
        dipole_au=np.array([0.1, -0.2, 0.3]),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
    )

    output = tmp_path / "water.EOut"
    write_external_output(output, request, result)
    parsed = parse_external_output(output, request)

    assert parsed.energy_hartree == pytest.approx(result.energy_hartree, abs=1e-12)
    assert np.allclose(parsed.dipole_au, result.dipole_au)
    assert np.allclose(parsed.gradient_hartree_per_bohr, gradient)
    assert np.allclose(parsed.hessian_hartree_per_bohr2, hessian)
    assert np.allclose(parsed.polarizability_au, 0.0)
    assert np.allclose(parsed.dipole_derivatives_au, 0.0)

    lines = output.read_text(encoding="ascii").splitlines()
    assert len(lines[0]) == 80  # 4D20.12
    assert "D" in lines[0]


def test_energy_only_has_one_line(tmp_path):
    source = tmp_path / "atom.EIn"
    source.write_text(
        "         1         0         0         1\n"
        "         1      0.000000000000      0.000000000000      0.000000000000      0.000000000000\n",
        encoding="ascii",
    )
    request = parse_external_input(source)
    output = tmp_path / "atom.EOut"
    write_external_output(output, request, ExternalResult(energy_hartree=-0.5))
    assert len(output.read_text(encoding="ascii").splitlines()) == 1


def test_rejects_missing_gradient(tmp_path):
    request = parse_external_input(FIXTURES / "water_deriv2.EIn")
    with pytest.raises(ExternalFormatError, match="requires a gradient"):
        write_external_output(
            tmp_path / "bad.EOut",
            request,
            ExternalResult(
                energy_hartree=-1.0,
                hessian_hartree_per_bohr2=np.eye(request.ndof),
            ),
        )


def test_rejects_asymmetric_hessian(tmp_path):
    request = parse_external_input(FIXTURES / "water_deriv2.EIn")
    bad_hessian = np.eye(request.ndof)
    bad_hessian[0, 1] = 0.1
    with pytest.raises(ExternalFormatError, match="not symmetric"):
        write_external_output(
            tmp_path / "bad.EOut",
            request,
            ExternalResult(
                energy_hartree=-1.0,
                gradient_hartree_per_bohr=np.zeros((request.natoms, 3)),
                hessian_hartree_per_bohr2=bad_hessian,
            ),
        )


def test_d_exponent_input(tmp_path):
    source = tmp_path / "dexp.EIn"
    source.write_text(
        "         1         0        -1         2\n"
        "         8  1.000000000000D+00  0.000000000000D+00  -2.000000000000D+00  0.000000000000D+00\n",
        encoding="ascii",
    )
    request = parse_external_input(source)
    assert request.charge == -1
    assert request.multiplicity == 2
    assert np.allclose(request.positions_bohr[0], [1.0, 0.0, -2.0])

"""Read and write Gaussian's External interface files.

The implementation is independent of MAPLE. It uses whitespace parsing for
input robustness and fixed-width D20.12 output for Gaussian compatibility.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np

from .errors import ExternalFormatError
from .models import ExternalRequest, ExternalResult


def _parse_float(token: str, *, context: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ExternalFormatError(
            f"Could not parse floating-point value {token!r} in {context}."
        ) from exc
    if not np.isfinite(value):
        raise ExternalFormatError(f"Non-finite value {token!r} in {context}.")
    return value


def _parse_int(token: str, *, context: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise ExternalFormatError(
            f"Could not parse integer value {token!r} in {context}."
        ) from exc


def parse_external_input(path: str | os.PathLike[str]) -> ExternalRequest:
    """Parse a Gaussian .EIn file.

    Gaussian normally writes four integer header fields. Any additional integer
    fields are retained in ``extra_header_fields`` for forward compatibility.
    """
    source = Path(path)
    try:
        with source.open("r", encoding="ascii") as handle:
            return _parse_external_input_stream(handle, source_path=source)
    except OSError as exc:
        raise ExternalFormatError(f"Could not read External input {source}: {exc}") from exc


def _parse_external_input_stream(
    handle: TextIO,
    *,
    source_path: Path | None = None,
) -> ExternalRequest:
    header_line = handle.readline()
    if not header_line:
        raise ExternalFormatError("Gaussian External input is empty.")

    header_tokens = header_line.split()
    if len(header_tokens) < 4:
        raise ExternalFormatError(
            "Gaussian External header must contain at least four integers: "
            "natoms, derivative order, charge, and multiplicity."
        )
    header = [
        _parse_int(token, context="External header") for token in header_tokens
    ]
    natoms, derivative_order, charge, multiplicity, *extra = header
    if natoms <= 0:
        raise ExternalFormatError(f"natoms must be positive, got {natoms}.")

    atomic_numbers = np.empty(natoms, dtype=np.int64)
    positions = np.empty((natoms, 3), dtype=np.float64)
    mm_charges = np.empty(natoms, dtype=np.float64)

    for index in range(natoms):
        line = handle.readline()
        if not line:
            raise ExternalFormatError(
                f"External input ended after {index} atom records; expected {natoms}."
            )
        tokens = line.split()
        if len(tokens) < 5:
            raise ExternalFormatError(
                f"Atom record {index + 1} must contain atomic number, x, y, z, "
                f"and MM charge; got {len(tokens)} fields."
            )
        atomic_numbers[index] = _parse_int(
            tokens[0],
            context=f"atom record {index + 1}",
        )
        positions[index] = [
            _parse_float(token, context=f"atom record {index + 1}")
            for token in tokens[1:4]
        ]
        mm_charges[index] = _parse_float(
            tokens[4],
            context=f"atom record {index + 1}",
        )

    # Gaussian 16 may append implementation-specific metadata after the
    # declared atom records (for example atom-type or layer bookkeeping).
    # The MLIP request is fully determined by the header and the first
    # ``natoms`` records, so tolerate and ignore any remaining non-empty
    # lines.  We still validate the complete required prefix strictly above:
    # a truncated or malformed atom block remains a hard error.
    for _line in handle:
        pass

    return ExternalRequest(
        atomic_numbers=atomic_numbers,
        positions_bohr=positions,
        derivative_order=derivative_order,
        charge=charge,
        multiplicity=multiplicity,
        mm_charges=mm_charges,
        extra_header_fields=tuple(extra),
        source_path=source_path,
    )


def _d20_12(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        raise ExternalFormatError("Cannot write NaN or infinity to Gaussian output.")
    field = f"{value:20.12E}".replace("E", "D")
    if len(field) != 20:
        raise ExternalFormatError(
            f"Value {value!r} does not fit Gaussian D20.12 format."
        )
    return field


def _write_values(handle: TextIO, values: Iterable[float], *, per_line: int) -> None:
    line: list[str] = []
    for value in values:
        line.append(_d20_12(float(value)))
        if len(line) == per_line:
            handle.write("".join(line) + "\n")
            line.clear()
    if line:
        handle.write("".join(line) + "\n")


def _lower_triangle_row_major(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    return np.fromiter(
        (matrix[i, j] for i in range(size) for j in range(i + 1)),
        dtype=np.float64,
        count=size * (size + 1) // 2,
    )


def write_external_output(
    path: str | os.PathLike[str],
    request: ExternalRequest,
    result: ExternalResult,
    *,
    atomic: bool = True,
) -> None:
    """Write a Gaussian .EOut file in fixed-width atomic-unit format."""
    target = Path(path)
    result.validated_for(request)
    target.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        temp_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(temp_handle.name)
        try:
            with temp_handle as handle:
                _write_external_output_stream(handle, request, result)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    else:
        try:
            with target.open("w", encoding="ascii", newline="\n") as handle:
                _write_external_output_stream(handle, request, result)
        except OSError as exc:
            raise ExternalFormatError(
                f"Could not write External output {target}: {exc}"
            ) from exc


def _write_external_output_stream(
    handle: TextIO,
    request: ExternalRequest,
    result: ExternalResult,
) -> None:
    # Energy and dipole moment: 4D20.12.
    _write_values(
        handle,
        [result.energy_hartree, *result.dipole_au],
        per_line=4,
    )

    if request.derivative_order >= 1:
        assert result.gradient_hartree_per_bohr is not None
        for row in result.gradient_hartree_per_bohr:
            _write_values(handle, row, per_line=3)

    if request.derivative_order == 2:
        assert result.polarizability_au is not None
        assert result.dipole_derivatives_au is not None
        assert result.hessian_hartree_per_bohr2 is not None

        # Polarizability: 6 values, 3D20.12.
        _write_values(handle, result.polarizability_au, per_line=3)
        # Dipole derivatives: 9*N values, 3D20.12.
        _write_values(handle, result.dipole_derivatives_au, per_line=3)
        # Cartesian force constants: lower triangle, row-major, 3D20.12.
        _write_values(
            handle,
            _lower_triangle_row_major(result.hessian_hartree_per_bohr2),
            per_line=3,
        )


def parse_external_output(
    path: str | os.PathLike[str],
    request: ExternalRequest,
) -> ExternalResult:
    """Parse Gau_MAPLE's own .EOut format for tests and diagnostics.

    This is not required by Gaussian at runtime, but makes protocol round-trip
    tests possible without relying on Gaussian itself.
    """
    source = Path(path)
    try:
        tokens = source.read_text(encoding="ascii").split()
    except OSError as exc:
        raise ExternalFormatError(f"Could not read External output {source}: {exc}") from exc

    values = np.asarray(
        [_parse_float(token, context="External output") for token in tokens],
        dtype=np.float64,
    )
    cursor = 0

    def take(count: int, *, section: str) -> np.ndarray:
        nonlocal cursor
        end = cursor + count
        if end > values.size:
            raise ExternalFormatError(
                f"External output is truncated in {section}: needed {count} more "
                f"values, found {values.size - cursor}."
            )
        chunk = values[cursor:end]
        cursor = end
        return chunk

    header = take(4, section="energy/dipole")
    energy = float(header[0])
    dipole = header[1:4].copy()

    gradient = None
    hessian = None
    polarizability = None
    dipole_derivatives = None

    if request.derivative_order >= 1:
        gradient = take(3 * request.natoms, section="gradient").reshape(
            request.natoms,
            3,
        )

    if request.derivative_order == 2:
        polarizability = take(6, section="polarizability").copy()
        dipole_derivatives = take(
            9 * request.natoms,
            section="dipole derivatives",
        ).copy()
        ntri = request.ndof * (request.ndof + 1) // 2
        triangle = take(ntri, section="Hessian")
        hessian = np.zeros((request.ndof, request.ndof), dtype=np.float64)
        tri_cursor = 0
        for i in range(request.ndof):
            for j in range(i + 1):
                value = triangle[tri_cursor]
                tri_cursor += 1
                hessian[i, j] = value
                hessian[j, i] = value

    if cursor != values.size:
        raise ExternalFormatError(
            f"External output contains {values.size - cursor} unexpected extra values."
        )

    return ExternalResult(
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        dipole_au=dipole,
        polarizability_au=polarizability,
        dipole_derivatives_au=dipole_derivatives,
    ).validated_for(request)

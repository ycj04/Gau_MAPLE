"""Safe framed protocol for local Gau_MAPLE server/client communication.

The transport is a Unix domain socket.  This is unrelated to SOCKS or HTTPS:
it is a local operating-system IPC endpoint represented by a filesystem path.

Messages are UTF-8 JSON compressed with zlib and preceded by a fixed binary
header.  No pickle or dynamic Python object deserialization is used.
"""

from __future__ import annotations

import json
import socket
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import ProtocolError
from .models import ExternalRequest, ExternalResult

PROTOCOL_VERSION = 2
_MAGIC = b"GMP4"
_HEADER = struct.Struct("!4sQ")
MAX_COMPRESSED_BYTES = 256 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


def _array_or_none(value: np.ndarray | None) -> list[Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    return array.tolist()


def request_to_payload(request: ExternalRequest) -> dict[str, Any]:
    return {
        "atomic_numbers": request.atomic_numbers.tolist(),
        "positions_bohr": request.positions_bohr.tolist(),
        "derivative_order": int(request.derivative_order),
        "charge": int(request.charge),
        "multiplicity": int(request.multiplicity),
        "mm_charges": request.mm_charges.tolist(),
        "extra_header_fields": list(request.extra_header_fields),
    }


def request_from_payload(payload: Mapping[str, Any]) -> ExternalRequest:
    try:
        return ExternalRequest(
            atomic_numbers=np.asarray(payload["atomic_numbers"], dtype=np.int64),
            positions_bohr=np.asarray(payload["positions_bohr"], dtype=np.float64),
            derivative_order=int(payload["derivative_order"]),
            charge=int(payload["charge"]),
            multiplicity=int(payload["multiplicity"]),
            mm_charges=np.asarray(payload["mm_charges"], dtype=np.float64),
            extra_header_fields=tuple(int(x) for x in payload.get("extra_header_fields", ())),
        )
    except KeyError as exc:
        raise ProtocolError(f"Request payload is missing field {exc.args[0]!r}.") from exc
    except Exception as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"Invalid request payload: {type(exc).__name__}: {exc}") from exc


def result_to_payload(result: ExternalResult) -> dict[str, Any]:
    return {
        "energy_hartree": float(result.energy_hartree),
        "gradient_hartree_per_bohr": _array_or_none(result.gradient_hartree_per_bohr),
        "hessian_hartree_per_bohr2": _array_or_none(result.hessian_hartree_per_bohr2),
        "dipole_au": _array_or_none(result.dipole_au),
        "polarizability_au": _array_or_none(result.polarizability_au),
        "dipole_derivatives_au": _array_or_none(result.dipole_derivatives_au),
    }


def result_from_payload(
    payload: Mapping[str, Any],
    request: ExternalRequest,
) -> ExternalResult:
    try:
        result = ExternalResult(
            energy_hartree=float(payload["energy_hartree"]),
            gradient_hartree_per_bohr=(
                None
                if payload.get("gradient_hartree_per_bohr") is None
                else np.asarray(payload["gradient_hartree_per_bohr"], dtype=np.float64)
            ),
            hessian_hartree_per_bohr2=(
                None
                if payload.get("hessian_hartree_per_bohr2") is None
                else np.asarray(payload["hessian_hartree_per_bohr2"], dtype=np.float64)
            ),
            dipole_au=np.asarray(payload.get("dipole_au", [0.0, 0.0, 0.0]), dtype=np.float64),
            polarizability_au=(
                None
                if payload.get("polarizability_au") is None
                else np.asarray(payload["polarizability_au"], dtype=np.float64)
            ),
            dipole_derivatives_au=(
                None
                if payload.get("dipole_derivatives_au") is None
                else np.asarray(payload["dipole_derivatives_au"], dtype=np.float64)
            ),
        )
        return result.validated_for(request)
    except KeyError as exc:
        raise ProtocolError(f"Result payload is missing field {exc.args[0]!r}.") from exc
    except Exception as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"Invalid result payload: {type(exc).__name__}: {exc}") from exc


def make_evaluate_message(
    request: ExternalRequest,
    profile_name: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "evaluate",
        "request": request_to_payload(request),
    }
    if profile_name is not None:
        message["profile"] = str(profile_name)
    return message


def make_ping_message(profile_name: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "ping",
    }
    if profile_name is not None:
        message["profile"] = str(profile_name)
    return message


def validate_message_envelope(message: Mapping[str, Any]) -> str:
    try:
        version = int(message["protocol_version"])
        message_type = str(message["type"])
    except KeyError as exc:
        raise ProtocolError(f"Protocol message is missing field {exc.args[0]!r}.") from exc
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported Gau_MAPLE protocol version {version}; expected {PROTOCOL_VERSION}."
        )
    return message_type


def encode_message(message: Mapping[str, Any]) -> bytes:
    try:
        raw = json.dumps(
            message,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Could not JSON-encode protocol message: {exc}") from exc
    if len(raw) > MAX_UNCOMPRESSED_BYTES:
        raise ProtocolError(
            f"Protocol message is too large before compression: {len(raw)} bytes."
        )
    compressed = zlib.compress(raw, level=3)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ProtocolError(
            f"Protocol message is too large after compression: {len(compressed)} bytes."
        )
    return _HEADER.pack(_MAGIC, len(compressed)) + compressed


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ProtocolError(
                f"Socket closed with {remaining} byte(s) still expected."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, message: Mapping[str, Any]) -> None:
    sock.sendall(encode_message(message))


def receive_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    magic, length = _HEADER.unpack(header)
    if magic != _MAGIC:
        raise ProtocolError("Invalid Gau_MAPLE protocol magic header.")
    if length > MAX_COMPRESSED_BYTES:
        raise ProtocolError(
            f"Incoming compressed frame is too large: {length} bytes."
        )
    compressed = _recv_exact(sock, int(length))
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, MAX_UNCOMPRESSED_BYTES + 1)
        if len(raw) > MAX_UNCOMPRESSED_BYTES or decompressor.unconsumed_tail:
            raise ProtocolError(
                "Incoming uncompressed protocol message exceeds the size limit."
            )
        raw += decompressor.flush(MAX_UNCOMPRESSED_BYTES + 1 - len(raw))
    except zlib.error as exc:
        raise ProtocolError(f"Could not decompress protocol frame: {exc}") from exc
    if len(raw) > MAX_UNCOMPRESSED_BYTES:
        raise ProtocolError("Incoming uncompressed protocol message exceeds the size limit.")
    try:
        message = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Could not decode protocol JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("Protocol message must be a JSON object.")
    return message


def normalize_socket_path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = result.absolute()
    # Linux sockaddr_un is typically limited to 108 bytes including terminator.
    if len(str(result).encode()) >= 104:
        raise ProtocolError(
            f"Unix socket path is too long ({len(str(result).encode())} bytes): {result}"
        )
    return result

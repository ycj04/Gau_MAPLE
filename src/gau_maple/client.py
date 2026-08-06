"""Client utilities for persistent multi-profile Gau_MAPLE servers."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import RemoteServerError, ServerConnectionError
from .models import ExternalRequest, ExternalResult
from .protocol import (
    make_evaluate_message,
    make_ping_message,
    normalize_socket_path,
    receive_message,
    result_from_payload,
    send_message,
    validate_message_envelope,
)


@dataclass(frozen=True, slots=True)
class ServerMetadata:
    server_name: str
    profile_name: str | None
    model: str | None
    device: str | None
    pid: int
    request_count: int
    available_profiles: tuple[str, ...] = ()
    profile_statuses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _metadata_from_response(response: dict[str, Any]) -> ServerMetadata:
    try:
        profile = response.get("profile")
        server = response["server"]
        raw_profiles = server.get("profiles", {})
        if not isinstance(raw_profiles, dict):
            raise TypeError("server.profiles must be an object")
        statuses = {
            str(name): MappingProxyType(dict(value))
            for name, value in raw_profiles.items()
            if isinstance(value, dict)
        }
        if profile is None:
            profile_name = model = device = None
        else:
            profile_name = str(profile["name"])
            model = str(profile["model"])
            device = str(profile["device"])
        return ServerMetadata(
            server_name=str(server.get("name", profile_name or "server")),
            profile_name=profile_name,
            model=model,
            device=device,
            pid=int(server["pid"]),
            request_count=int(server.get("request_count", 0)),
            available_profiles=tuple(sorted(statuses)),
            profile_statuses=MappingProxyType(statuses),
        )
    except Exception as exc:
        raise RemoteServerError(
            f"Server response contains invalid identity metadata: {exc}"
        ) from exc


def _exchange(
    socket_path: str | Path,
    message: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    path = normalize_socket_path(socket_path)
    if timeout <= 0:
        raise ServerConnectionError("Socket timeout must be positive.")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(float(timeout))
    try:
        sock.connect(str(path))
        send_message(sock, message)
        return receive_message(sock)
    except (OSError, TimeoutError) as exc:
        raise ServerConnectionError(
            f"Could not communicate with Gau_MAPLE server at {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sock.close()


def _raise_remote_error(response: dict[str, Any]) -> None:
    error = response.get("error")
    if not isinstance(error, dict):
        raise RemoteServerError("Server returned an error response without diagnostics.")
    remote_type = str(error.get("type", "RemoteError"))
    message = str(error.get("message", "unknown server error"))
    remote_traceback = str(error.get("traceback", ""))
    detail = f"{remote_type}: {message}"
    if remote_traceback:
        detail += f"\n--- remote traceback ---\n{remote_traceback}"
    raise RemoteServerError(detail)


def _validate_identity(
    metadata: ServerMetadata,
    *,
    expect_server: str | None,
    expect_profile: str | None,
) -> None:
    if expect_server and metadata.server_name != expect_server:
        raise RemoteServerError(
            f"Connected to server {metadata.server_name!r}, expected {expect_server!r}."
        )
    if expect_profile and metadata.profile_name != expect_profile:
        raise RemoteServerError(
            f"Connected to profile {metadata.profile_name!r}, expected {expect_profile!r}."
        )


def ping_server(
    socket_path: str | Path,
    *,
    timeout: float = 10.0,
    profile_name: str | None = None,
    expect_server: str | None = None,
    expect_profile: str | None = None,
) -> ServerMetadata:
    selected_profile = profile_name or expect_profile
    response = _exchange(
        socket_path,
        make_ping_message(selected_profile),
        timeout=timeout,
    )
    response_type = validate_message_envelope(response)
    if response_type == "error":
        _raise_remote_error(response)
    if response_type != "pong":
        raise RemoteServerError(f"Expected pong response, received {response_type!r}.")
    metadata = _metadata_from_response(response)
    _validate_identity(
        metadata,
        expect_server=expect_server,
        expect_profile=expect_profile,
    )
    return metadata


def evaluate_via_server(
    request: ExternalRequest,
    socket_path: str | Path,
    *,
    profile_name: str | None = None,
    timeout: float = 600.0,
    expect_server: str | None = None,
    expect_profile: str | None = None,
) -> tuple[ExternalResult, ServerMetadata]:
    selected_profile = profile_name or expect_profile
    response = _exchange(
        socket_path,
        make_evaluate_message(request, selected_profile),
        timeout=timeout,
    )
    response_type = validate_message_envelope(response)
    if response_type == "error":
        _raise_remote_error(response)
    if response_type != "result":
        raise RemoteServerError(f"Expected result response, received {response_type!r}.")
    metadata = _metadata_from_response(response)
    _validate_identity(
        metadata,
        expect_server=expect_server,
        expect_profile=expect_profile,
    )
    payload = response.get("result")
    if not isinstance(payload, dict):
        raise RemoteServerError("Server result response is missing a result object.")
    return result_from_payload(payload, request), metadata

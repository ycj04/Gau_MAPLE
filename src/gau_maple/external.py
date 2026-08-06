"""Executable Gaussian External bridge for direct and persistent-server modes."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .cli import (
    build_parser,
    profile_from_namespace,
    remote_profile_from_namespace,
    resolve_config_client,
    validate_external_mode,
)
from .client import evaluate_via_server
from .gaussian_io import parse_external_input, write_external_output
from .invocation import GaussianInvocation, parse_gaussian_invocation
from .maple_backend import MapleBackend
from .profiles import MapleProfile


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reset_message_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_message(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def _profile_summary(profile: MapleProfile) -> str:
    options = ", ".join(
        f"{key}={value!r}" for key, value in sorted(profile.model_options.items())
    ) or "(none)"
    return (
        f"profile={profile.name!r}; model={profile.model!r}; "
        f"device={profile.device!r}; model_options={options}"
    )


def _prepare_invocation(invocation: GaussianInvocation) -> None:
    invocation.validate_direct_mode()
    invocation.output_path.parent.mkdir(parents=True, exist_ok=True)
    invocation.output_path.unlink(missing_ok=True)
    _reset_message_file(invocation.message_path)


def run_direct_external(
    invocation: GaussianInvocation,
    profile: MapleProfile,
    *,
    backend_log: str | Path | None = None,
    backend_factory: Callable[..., MapleBackend] = MapleBackend,
) -> None:
    """Execute one validated direct-mode request."""
    _prepare_invocation(invocation)
    log_path = Path(backend_log).expanduser() if backend_log else invocation.message_path
    _append_message(
        invocation.message_path,
        f"[Gau_MAPLE] {_utc_timestamp()} direct request started",
    )
    _append_message(invocation.message_path, f"[Gau_MAPLE] {_profile_summary(profile)}")

    request = parse_external_input(invocation.input_path)
    backend = backend_factory(profile, log_path=log_path)
    result = backend.evaluate(request)
    write_external_output(invocation.output_path, request, result, atomic=True)

    _append_message(
        invocation.message_path,
        "[Gau_MAPLE] completed successfully; "
        f"mode=direct; natoms={request.natoms}; "
        f"derivative_order={request.derivative_order}; "
        f"energy_hartree={result.energy_hartree:.12f}",
    )


def run_socket_external(
    invocation: GaussianInvocation,
    socket_path: str | Path,
    *,
    profile_name: str | None = None,
    timeout: float = 600.0,
    expect_server: str | None = None,
    expect_profile: str | None = None,
) -> None:
    """Execute one request through a persistent local Gau_MAPLE server."""
    _prepare_invocation(invocation)
    _append_message(
        invocation.message_path,
        f"[Gau_MAPLE] {_utc_timestamp()} socket request started; socket={socket_path}",
    )
    request = parse_external_input(invocation.input_path)
    result, metadata = evaluate_via_server(
        request,
        socket_path,
        profile_name=profile_name,
        timeout=timeout,
        expect_server=expect_server,
        expect_profile=expect_profile,
    )
    write_external_output(invocation.output_path, request, result, atomic=True)
    _append_message(
        invocation.message_path,
        "[Gau_MAPLE] completed successfully; "
        f"mode=socket; server={metadata.server_name!r}; server_pid={metadata.pid}; "
        f"profile={metadata.profile_name!r}; model={metadata.model!r}; "
        f"device={metadata.device!r}; request_count={metadata.request_count}; "
        f"natoms={request.natoms}; derivative_order={request.derivative_order}; "
        f"energy_hartree={result.energy_hartree:.12f}",
    )


def _write_failure(invocation: GaussianInvocation | None, exc: BaseException) -> None:
    diagnostic = (
        f"[Gau_MAPLE ERROR] {_utc_timestamp()} {type(exc).__name__}: {exc}\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    if invocation is not None:
        try:
            invocation.output_path.unlink(missing_ok=True)
            _append_message(invocation.message_path, diagnostic)
        except Exception:
            pass
    print(f"Gau_MAPLE failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv in (["--help"], ["-h"]):
        parser = build_parser()
        parser.print_help()
        print(
            "\nRuntime form: gau-maple [OPTIONS] "
            "layer InputFile OutputFile MsgFile FChkFile MatElFile"
        )
        return 0
    if raw_argv == ["--version"]:
        from . import __version__

        print(__version__)
        return 0

    invocation: GaussianInvocation | None = None
    debug = False
    try:
        invocation = parse_gaussian_invocation(raw_argv)
        parser = build_parser()
        args = parser.parse_args(list(invocation.option_argv))
        debug = bool(args.debug)
        resolve_config_client(args)
        mode = validate_external_mode(args)
        if mode == "socket":
            remote_profile = remote_profile_from_namespace(args)
            run_socket_external(
                invocation,
                args.socket,
                profile_name=remote_profile,
                timeout=float(args.timeout),
                expect_server=args.expect_server,
                expect_profile=remote_profile,
            )
        else:
            profile = profile_from_namespace(args)
            run_direct_external(
                invocation,
                profile,
                backend_log=args.backend_log,
            )
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        _write_failure(invocation, exc)
        if debug:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

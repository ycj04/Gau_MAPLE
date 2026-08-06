"""Persistent multi-profile Gau_MAPLE server over a Unix domain socket."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import socketserver
import stat
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .cli import add_profile_arguments, profile_from_namespace
from .config import default_config_path, load_config
from .errors import ConfigError, ProfileError, ProtocolError, ServerStartupError
from .maple_backend import MapleBackend
from .models import ExternalRequest
from .profiles import MapleProfile
from .protocol import (
    PROTOCOL_VERSION,
    normalize_socket_path,
    receive_message,
    request_from_payload,
    result_to_payload,
    send_message,
    validate_message_envelope,
)
from .server_groups import get_server_group, validate_profile_mapping


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def _profile_payload(profile: MapleProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "model": profile.model,
        "device": profile.device,
    }


def _warmup_request() -> ExternalRequest:
    # Neutral singlet water in Bohr.  A derivative-order-zero call is enough to
    # force checkpoint/model construction while minimizing startup cost.
    ang_to_bohr = 1.0 / 0.529177210903
    positions_ang = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [0.757160, 0.000000, 0.586260],
            [-0.757160, 0.000000, 0.586260],
        ],
        dtype=np.float64,
    )
    return ExternalRequest(
        atomic_numbers=np.array([8, 1, 1], dtype=np.int64),
        positions_bohr=positions_ang * ang_to_bohr,
        derivative_order=0,
        charge=0,
        multiplicity=1,
        mm_charges=np.zeros(3, dtype=np.float64),
        extra_header_fields=(),
    )


@dataclass(slots=True)
class _ProfileRuntime:
    profile: MapleProfile
    backend: MapleBackend
    lock: threading.Lock
    request_count: int = 0
    preload_state: str = "lazy"
    preload_error: str | None = None

    def payload(self) -> dict[str, Any]:
        result = _profile_payload(self.profile)
        result.update(
            {
                "request_count": self.request_count,
                "preload_state": self.preload_state,
                "preload_error": self.preload_error,
            }
        )
        return result


@dataclass(slots=True)
class _ServerState:
    server_name: str
    profiles: dict[str, _ProfileRuntime]
    log_path: Path
    request_count: int = 0
    count_lock: threading.Lock = field(default_factory=threading.Lock)

    def selected_runtime(self, profile_name: str | None) -> _ProfileRuntime | None:
        if profile_name is None:
            return None
        return self.profiles.get(profile_name)

    def metadata(self, selected_profile: str | None = None) -> dict[str, Any]:
        runtime = self.selected_runtime(selected_profile)
        return {
            "profile": None if runtime is None else runtime.payload(),
            "server": {
                "name": self.server_name,
                "pid": os.getpid(),
                "request_count": self.request_count,
                "profiles": {
                    name: runtime.payload()
                    for name, runtime in sorted(self.profiles.items())
                },
            },
        }


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: GauMapleMultiProfileServer = self.server  # type: ignore[assignment]
        selected_profile: str | None = None
        try:
            message = receive_message(self.request)
            raw_profile = message.get("profile")
            selected_profile = None if raw_profile is None else str(raw_profile)
            response = server.dispatch(message)
        except Exception as exc:
            response = server.error_response(exc, selected_profile=selected_profile)
        try:
            send_message(self.request, response)
        except Exception as exc:
            _append_log(
                server.state.log_path,
                f"[Gau_MAPLE SERVER] {_utc_timestamp()} failed to send response: "
                f"{type(exc).__name__}: {exc}",
            )


class GauMapleMultiProfileServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """One process serving multiple cached MAPLE calculator profiles.

    Connections may be accepted concurrently, but each profile has its own
    lock because ASE calculators mutate ``atoms`` and ``results``.  Different
    profiles may therefore evaluate concurrently while requests targeting the
    same calculator remain serialized.
    """

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        socket_path: str | Path,
        profiles: Mapping[str, MapleProfile],
        *,
        server_name: str,
        log_path: str | Path,
        backend_factory: Callable[..., MapleBackend] = MapleBackend,
        preload: bool = False,
    ) -> None:
        self.socket_path = normalize_socket_path(socket_path)
        self.log_path = Path(log_path).expanduser().absolute()
        cleaned_profiles = validate_profile_mapping(profiles)
        runtimes = {
            name: _ProfileRuntime(
                profile=profile,
                backend=backend_factory(profile, log_path=self.log_path),
                lock=threading.Lock(),
                preload_state="pending" if preload else "lazy",
            )
            for name, profile in cleaned_profiles.items()
        }
        self.state = _ServerState(
            server_name=str(server_name).strip(),
            profiles=runtimes,
            log_path=self.log_path,
        )
        if not self.state.server_name:
            raise ProfileError("server_name must not be empty.")

        if preload:
            self.preload_all()

        self._prepare_socket_path(self.socket_path)
        super().__init__(str(self.socket_path), _RequestHandler)
        os.chmod(self.socket_path, 0o600)

    @staticmethod
    def _prepare_socket_path(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        if not path.exists() and not path.is_symlink():
            return
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ServerStartupError(f"Could not inspect existing socket path {path}: {exc}") from exc
        if not stat.S_ISSOCK(mode):
            raise ServerStartupError(f"Refusing to replace existing non-socket path: {path}")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(path))
        except OSError:
            path.unlink(missing_ok=True)
        else:
            raise ServerStartupError(f"A Gau_MAPLE server is already listening at {path}.")
        finally:
            probe.close()

    def preload_all(self) -> None:
        request = _warmup_request()
        for name, runtime in self.state.profiles.items():
            runtime.preload_state = "loading"
            runtime.preload_error = None
            _append_log(
                self.state.log_path,
                f"[Gau_MAPLE PRELOAD] {_utc_timestamp()} profile={name!r} started",
            )
            try:
                with runtime.lock:
                    runtime.backend.evaluate(request)
                runtime.preload_state = "loaded"
                _append_log(
                    self.state.log_path,
                    f"[Gau_MAPLE PRELOAD] {_utc_timestamp()} profile={name!r} loaded",
                )
            except Exception as exc:
                runtime.preload_state = "failed"
                runtime.preload_error = f"{type(exc).__name__}: {exc}"
                _append_log(
                    self.state.log_path,
                    f"[Gau_MAPLE PRELOAD ERROR] {_utc_timestamp()} profile={name!r}: "
                    f"{runtime.preload_error}\n"
                    + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )

    def _resolve_profile(self, raw_name: Any) -> tuple[str, _ProfileRuntime]:
        if raw_name is None:
            if len(self.state.profiles) == 1:
                name = next(iter(self.state.profiles))
                return name, self.state.profiles[name]
            available = ", ".join(sorted(self.state.profiles))
            raise ProtocolError(
                "This server contains multiple profiles; the client must specify "
                f"one of: {available}."
            )
        name = str(raw_name).strip()
        runtime = self.state.profiles.get(name)
        if runtime is None:
            available = ", ".join(sorted(self.state.profiles))
            raise ProtocolError(
                f"Unknown server profile {name!r}; available profiles: {available}."
            )
        return name, runtime

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.socket_path.unlink(missing_ok=True)

    def error_response(
        self,
        exc: BaseException,
        *,
        selected_profile: str | None = None,
    ) -> dict[str, Any]:
        remote_traceback = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        _append_log(
            self.state.log_path,
            f"[Gau_MAPLE SERVER ERROR] {_utc_timestamp()} "
            f"{type(exc).__name__}: {exc}\n{remote_traceback}",
        )
        response: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "type": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": remote_traceback,
            },
        }
        response.update(self.state.metadata(selected_profile))
        return response

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        message_type = validate_message_envelope(message)
        raw_profile = message.get("profile")
        if message_type == "ping":
            selected_name: str | None = None
            if raw_profile is not None:
                selected_name, _ = self._resolve_profile(raw_profile)
            response: dict[str, Any] = {
                "protocol_version": PROTOCOL_VERSION,
                "type": "pong",
            }
            response.update(self.state.metadata(selected_name))
            return response
        if message_type != "evaluate":
            raise ProtocolError(f"Unsupported server request type {message_type!r}.")

        profile_name, runtime = self._resolve_profile(raw_profile)
        payload = message.get("request")
        if not isinstance(payload, dict):
            raise ProtocolError("Evaluate message is missing a request object.")
        request = request_from_payload(payload)

        with runtime.lock:
            result = runtime.backend.evaluate(request)
            runtime.request_count += 1
            runtime.preload_state = "loaded"
            runtime.preload_error = None

        with self.state.count_lock:
            self.state.request_count += 1

        response = {
            "protocol_version": PROTOCOL_VERSION,
            "type": "result",
            "result": result_to_payload(result),
        }
        response.update(self.state.metadata(profile_name))
        return response


class GauMapleUnixServer(GauMapleMultiProfileServer):
    """Backward-compatible single-profile constructor used by older tests/code."""

    def __init__(
        self,
        socket_path: str | Path,
        profile: MapleProfile,
        *,
        log_path: str | Path,
        backend_factory: Callable[..., MapleBackend] = MapleBackend,
        preload: bool = False,
    ) -> None:
        super().__init__(
            socket_path,
            {profile.name: profile},
            server_name=profile.name,
            log_path=log_path,
            backend_factory=backend_factory,
            preload=preload,
        )


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gau-maple-server",
        description=(
            "Serve profiles from a Gau_MAPLE TOML config, a legacy built-in "
            "group, or one explicit MAPLE profile over a local Unix domain socket."
        ),
        allow_abbrev=False,
    )
    add_profile_arguments(parser, model_required=False)
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML configuration file",
    )
    parser.add_argument(
        "--server",
        help="server name inside --config, e.g. maple_server or meta_server",
    )
    parser.add_argument(
        "--group",
        choices=("maple", "meta"),
        help="legacy built-in multi-profile group",
    )
    parser.add_argument(
        "--server-name",
        help="legacy server identity override",
    )
    preload_group = parser.add_mutually_exclusive_group()
    preload_group.add_argument(
        "--preload",
        dest="preload",
        action="store_true",
        help="eagerly load all profiles before opening the socket",
    )
    preload_group.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="load profiles lazily on first use",
    )
    parser.set_defaults(preload=None)
    parser.add_argument(
        "--socket",
        type=Path,
        help="legacy explicit Unix socket path; supplied by TOML in config mode",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="legacy explicit server/backend log; supplied by TOML in config mode",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="serve one connection and exit; intended for diagnostics",
    )
    return parser


def _legacy_profiles_from_args(
    args: argparse.Namespace,
) -> tuple[str, dict[str, MapleProfile]]:
    if args.group:
        direct_fields = [
            args.model,
            args.module,
            args.option,
            args.d4,
            None if args.implicit == "none" else args.implicit,
            None if args.solvent == "none" else args.solvent,
            args.solvation_option,
            args.allow_unsupported_charge_mult,
        ]
        if any(direct_fields):
            raise ProfileError(
                "--group cannot be combined with --model/--module/--option or "
                "other single-profile construction options."
            )
        default_name, profiles = get_server_group(args.group)
        return args.server_name or default_name, profiles

    if not args.model:
        raise ProfileError(
            "Specify --config/--server, --group maple|meta, or construct one "
            "legacy profile with --model."
        )
    profile = profile_from_namespace(args)
    return args.server_name or profile.name, {profile.name: profile}


def run_server(
    profiles: Mapping[str, MapleProfile],
    socket_path: str | Path,
    *,
    server_name: str,
    log_path: str | Path | None = None,
    once: bool = False,
    preload: bool = False,
    backend_factory: Callable[..., MapleBackend] = MapleBackend,
) -> None:
    socket_path = normalize_socket_path(socket_path)
    log = (
        Path(log_path).expanduser().absolute()
        if log_path is not None
        else socket_path.with_suffix(socket_path.suffix + ".server.log")
    )
    server = GauMapleMultiProfileServer(
        socket_path,
        profiles,
        server_name=server_name,
        log_path=log,
        backend_factory=backend_factory,
        preload=preload,
    )
    _append_log(
        log,
        f"[Gau_MAPLE SERVER] {_utc_timestamp()} started; pid={os.getpid()}; "
        f"socket={socket_path}; server={server_name}; "
        f"profiles={','.join(sorted(profiles))}; preload={preload}",
    )

    def request_shutdown(signum: int, frame: Any) -> None:
        _append_log(log, f"[Gau_MAPLE SERVER] received signal {signum}; shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, request_shutdown)
        except ValueError:
            pass

    try:
        if once:
            server.handle_request()
        else:
            server.serve_forever(poll_interval=0.2)
    finally:
        if not once:
            server.shutdown()
        server.server_close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _append_log(log, f"[Gau_MAPLE SERVER] {_utc_timestamp()} stopped")


def _reject_config_legacy_mix(args: argparse.Namespace) -> None:
    conflicts = {
        "--group": args.group,
        "--server-name": args.server_name,
        "--model": args.model,
        "--module": args.module,
        "--option": args.option,
        "--d4": args.d4,
        "--implicit": None if args.implicit == "none" else args.implicit,
        "--solvent": None if args.solvent == "none" else args.solvent,
        "--solvation-option": args.solvation_option,
        "--socket": args.socket,
        "--log": args.log,
    }
    supplied = [key for key, value in conflicts.items() if value]
    if supplied:
        raise ConfigError(
            "Config mode takes socket/log/profile definitions from TOML; remove "
            f"legacy option(s): {', '.join(supplied)}."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_server_parser()
    args = parser.parse_args(argv)
    try:
        if args.config is not None or args.server is not None:
            if not args.server:
                raise ConfigError("Config mode requires --server SERVER_NAME.")
            _reject_config_legacy_mix(args)
            config_path = default_config_path(args.config)
            config = load_config(config_path)
            definition = config.get_server(args.server)
            # Apply the server-specific environment inside the process as well as
            # in gau-maple-ctl.  This makes manual launches robust against Conda
            # activation hooks such as MAPLE_CALCULATOR_PLUGINS=maple_mace_native.
            for key, value in definition.environment.items():
                os.environ[key] = value
            profiles = config.profiles_for_server(definition.name)
            preload = definition.preload if args.preload is None else bool(args.preload)
            run_server(
                profiles,
                definition.socket_path,
                server_name=definition.name,
                log_path=definition.log_path,
                once=bool(args.once),
                preload=preload,
            )
        else:
            server_name, profiles = _legacy_profiles_from_args(args)
            if args.socket is None:
                raise ProfileError("Legacy server mode requires --socket PATH.")
            run_server(
                profiles,
                args.socket,
                server_name=server_name,
                log_path=args.log,
                once=bool(args.once),
                preload=bool(args.preload),
            )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"gau-maple-server failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

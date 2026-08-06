"""Unified lifecycle manager for TOML-defined Gau_MAPLE servers."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

from .client import ServerMetadata, ping_server
from .config import (
    GauMapleConfig,
    ServerDefinition,
    default_config_path,
    load_config,
    validate_runtime,
)
from .errors import ConfigError, ServerConnectionError


def build_child_environment(
    overrides: Mapping[str, str],
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a deterministic child environment.

    An explicit empty value is retained.  This matters for
    ``MAPLE_CALCULATOR_PLUGINS``: setting it to ``""`` neutralizes activation
    hooks inherited from another Conda environment without relying on the
    interactive shell to run ``unset`` first.
    """

    env = dict(os.environ if base is None else base)
    for raw_key, raw_value in overrides.items():
        key = str(raw_key).strip()
        if not key or "=" in key or "\x00" in key:
            raise ConfigError(f"Invalid environment variable name: {raw_key!r}.")
        value = str(raw_value)
        if "\x00" in value:
            raise ConfigError(f"Environment value for {key!r} contains NUL.")
        env[key] = value
    return env


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid > 1 else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _server_cmdline_matches(pid: int, server_name: str) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    fields = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    joined = " ".join(fields)
    return "gau-maple-server" in joined and server_name in joined


def _ping(definition: ServerDefinition, *, timeout: float = 2.0) -> ServerMetadata:
    return ping_server(
        definition.socket_path,
        timeout=timeout,
        expect_server=definition.name,
    )


def _format_metadata(metadata: ServerMetadata) -> list[str]:
    lines = [
        f"server={metadata.server_name} pid={metadata.pid} "
        f"requests={metadata.request_count}",
        f"profiles={','.join(metadata.available_profiles)}",
    ]
    for name in metadata.available_profiles:
        status = metadata.profile_statuses[name]
        line = (
            f"  {name}: state={status.get('preload_state')} "
            f"requests={status.get('request_count', 0)}"
        )
        if status.get("preload_error"):
            line += f" error={status['preload_error']}"
        lines.append(line)
    return lines


def start_server(config: GauMapleConfig, definition: ServerDefinition) -> bool:
    """Start one server and wait for its socket. Return False if degraded/failed."""

    try:
        metadata = _ping(definition)
    except Exception:
        metadata = None
    if metadata is not None:
        print(f"{definition.name}: already running")
        for line in _format_metadata(metadata):
            print(line)
        return not any(
            status.get("preload_state") == "failed"
            for status in metadata.profile_statuses.values()
        )

    old_pid = _read_pid(definition.pid_file)
    if old_pid is not None and _pid_alive(old_pid):
        if _server_cmdline_matches(old_pid, definition.name):
            raise ConfigError(
                f"{definition.name}: PID {old_pid} is alive but its socket is not "
                f"responding; inspect {definition.stdout_path} before restarting."
            )
        raise ConfigError(
            f"{definition.name}: stale pid file points to unrelated live PID {old_pid}; "
            f"refusing to kill or overwrite it: {definition.pid_file}."
        )
    definition.pid_file.unlink(missing_ok=True)

    for path in (
        config.runtime_dir,
        definition.socket_path.parent,
        definition.pid_file.parent,
        definition.log_path.parent,
        definition.stdout_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass

    if not definition.executable.is_file() or not os.access(definition.executable, os.X_OK):
        raise ConfigError(
            f"{definition.name}: server executable is missing or not executable: "
            f"{definition.executable}. Install Gau_MAPLE in that Conda environment."
        )

    environment = build_child_environment(definition.environment)
    command = [
        str(definition.executable),
        "--config",
        str(config.source_path),
        "--server",
        definition.name,
    ]
    with definition.stdout_path.open("ab", buffering=0) as output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    _atomic_write_text(definition.pid_file, f"{process.pid}\n")
    print(
        f"{definition.name}: started PID {process.pid}; "
        f"waiting up to {definition.startup_timeout:g} s"
    )

    deadline = time.monotonic() + definition.startup_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            definition.pid_file.unlink(missing_ok=True)
            raise ConfigError(
                f"{definition.name}: server exited during startup with code {code}; "
                f"see {definition.stdout_path} and {definition.log_path}."
            )
        try:
            metadata = _ping(definition, timeout=2.0)
            print(f"{definition.name}: READY socket={definition.socket_path}")
            for line in _format_metadata(metadata):
                print(line)
            failed = [
                name
                for name, status in metadata.profile_statuses.items()
                if status.get("preload_state") == "failed"
            ]
            if failed:
                print(
                    f"{definition.name}: DEGRADED; failed profiles: {', '.join(failed)}",
                    file=sys.stderr,
                )
                return False
            return True
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)

    raise ConfigError(
        f"{definition.name}: did not become ready within "
        f"{definition.startup_timeout:g} s; last probe error: {last_error}. "
        f"See {definition.stdout_path} and {definition.log_path}."
    )


def stop_server(definition: ServerDefinition) -> bool:
    """Stop one server conservatively, avoiding unrelated stale PIDs."""

    metadata: ServerMetadata | None
    try:
        metadata = _ping(definition)
    except Exception:
        metadata = None

    pid_file_pid = _read_pid(definition.pid_file)
    if metadata is not None:
        pid = metadata.pid
        if pid_file_pid is not None and pid_file_pid != pid:
            raise ConfigError(
                f"{definition.name}: pid file contains {pid_file_pid}, but socket "
                f"reports {pid}; refusing ambiguous shutdown."
            )
    else:
        pid = pid_file_pid

    if pid is None or not _pid_alive(pid):
        definition.pid_file.unlink(missing_ok=True)
        definition.socket_path.unlink(missing_ok=True)
        print(f"{definition.name}: already stopped")
        return True

    if metadata is None and not _server_cmdline_matches(pid, definition.name):
        raise ConfigError(
            f"{definition.name}: PID {pid} is alive but cannot be verified as its "
            "gau-maple-server; refusing to signal it."
        )

    print(f"{definition.name}: sending SIGTERM to PID {pid}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + definition.shutdown_timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.2)
    else:
        print(
            f"{definition.name}: did not stop in {definition.shutdown_timeout:g} s; "
            "sending SIGKILL",
            file=sys.stderr,
        )
        os.kill(pid, signal.SIGKILL)
        for _ in range(50):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)

    definition.pid_file.unlink(missing_ok=True)
    definition.socket_path.unlink(missing_ok=True)
    print(f"{definition.name}: STOPPED")
    return not _pid_alive(pid)


def status_server(definition: ServerDefinition) -> bool:
    try:
        metadata = _ping(definition)
    except Exception as exc:
        pid = _read_pid(definition.pid_file)
        detail = f" pid_file={pid}" if pid is not None else ""
        print(f"{definition.name}: DOWN{detail}; {type(exc).__name__}: {exc}")
        return False
    print(f"{definition.name}: UP socket={definition.socket_path}")
    for line in _format_metadata(metadata):
        print(line)
    return not any(
        status.get("preload_state") == "failed"
        for status in metadata.profile_statuses.values()
    )


def _targets(config: GauMapleConfig, target: str) -> list[ServerDefinition]:
    if target == "all":
        return [config.servers[name] for name in sorted(config.servers)]
    return [config.get_server(target)]


def _tail(path: Path, lines: int) -> str:
    if not path.is_file():
        return f"<missing: {path}>"
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gau-maple-ctl", allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="profiles.toml path")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in ("start", "stop", "restart", "status"):
        child = sub.add_parser(command)
        child.add_argument("target", nargs="?", default="all")

    sub.add_parser("validate")
    sub.add_parser("list")
    logs = sub.add_parser("logs")
    logs.add_argument("target")
    logs.add_argument("--lines", type=int, default=80)
    logs.add_argument("--stdout", action="store_true", help="show launcher stdout log")
    logs.add_argument("--follow", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path = default_config_path(args.config)
        config = load_config(config_path)

        if args.command == "validate":
            print(f"PASS config={config.source_path}")
            print(f"runtime_dir={config.runtime_dir}")
            for line in validate_runtime(config):
                print(line)
            return 0

        if args.command == "list":
            print(f"config={config.source_path}")
            for name in sorted(config.servers):
                server = config.servers[name]
                print(
                    f"{name}: executable={server.executable} "
                    f"socket={server.socket_path} preload={server.preload}"
                )
                print(f"  profiles={','.join(server.profiles)}")
                if server.environment:
                    rendered = ", ".join(
                        f"{key}={value!r}" for key, value in server.environment.items()
                    )
                    print(f"  environment={rendered}")
            return 0

        if args.command == "logs":
            definition = config.get_server(args.target)
            path = definition.stdout_path if args.stdout else definition.log_path
            if args.follow:
                return subprocess.call(["tail", "-n", str(args.lines), "-F", str(path)])
            print(_tail(path, args.lines))
            return 0

        targets = _targets(config, args.target)
        results: list[bool] = []
        if args.command == "start":
            results = [start_server(config, definition) for definition in targets]
        elif args.command == "stop":
            results = [stop_server(definition) for definition in reversed(targets)]
        elif args.command == "restart":
            for definition in reversed(targets):
                stop_server(definition)
            results = [start_server(config, definition) for definition in targets]
        elif args.command == "status":
            results = [status_server(definition) for definition in targets]
        else:  # pragma: no cover
            parser.error(f"Unknown command: {args.command}")
        return 0 if all(results) else 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"gau-maple-ctl failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

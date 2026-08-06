"""Installation and runtime diagnostics for Gau_MAPLE.

The doctor command is intentionally read-only.  It validates the TOML file,
checks both Conda-owned server executables, inspects profile-specific model
paths, optionally probes running Unix-domain-socket servers, and can verify a
Gaussian executable.  It never starts or stops a server and never loads an
MLIP checkpoint in the calling process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import __version__
from .client import ping_server
from .config import GauMapleConfig, ServerDefinition, default_config_path, load_config
from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class Diagnostic:
    name: str
    status: str
    message: str
    detail: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    @property
    def warned(self) -> bool:
        return self.status == "WARN"


def _pass(name: str, message: str, detail: str | None = None) -> Diagnostic:
    return Diagnostic(name=name, status="PASS", message=message, detail=detail)


def _warn(name: str, message: str, detail: str | None = None) -> Diagnostic:
    return Diagnostic(name=name, status="WARN", message=message, detail=detail)


def _fail(name: str, message: str, detail: str | None = None) -> Diagnostic:
    return Diagnostic(name=name, status="FAIL", message=message, detail=detail)


def _python_for_server(definition: ServerDefinition) -> Path:
    """Infer the Python executable paired with a console-script executable."""

    return definition.executable.parent / "python"


def _run_import_probe(
    python_executable: Path,
    modules: Sequence[str],
    *,
    environment: dict[str, str],
    timeout: float,
) -> tuple[bool, str]:
    unique = tuple(dict.fromkeys(module for module in modules if module))
    code = (
        "import importlib, json, sys\n"
        f"mods={unique!r}\n"
        "out={}\n"
        "for name in mods:\n"
        "    mod=importlib.import_module(name)\n"
        "    out[name]=getattr(mod, '__file__', '<built-in>')\n"
        "print(json.dumps({'python': sys.executable, 'modules': out}, sort_keys=True))\n"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = error or output or f"return code {completed.returncode}"
        return False, detail
    return True, output


def _profile_modules(config: GauMapleConfig, definition: ServerDefinition) -> tuple[str, ...]:
    modules = ["gau_maple", "maple"]
    for profile_name in definition.profiles:
        profile = config.profiles[profile_name]
        module = profile.model_options.get("module")
        if module:
            modules.append(str(module))
    return tuple(dict.fromkeys(modules))


def _child_environment(definition: ServerDefinition) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in definition.environment.items():
        env[str(key)] = str(value)
    return env


def _check_python() -> list[Diagnostic]:
    diagnostics = [
        _pass(
            "python",
            f"Python {platform.python_version()} at {sys.executable}",
            detail=f"platform={platform.platform()}",
        ),
        _pass("gau_maple", f"Gau_MAPLE {__version__}"),
    ]
    if sys.version_info < (3, 10):
        diagnostics[0] = _fail("python", "Python >=3.10 is required.")
    return diagnostics


def _check_current_imports() -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for module in ("numpy", "ase"):
        spec = importlib.util.find_spec(module)
        if spec is None:
            diagnostics.append(_fail(f"import:{module}", f"Cannot import {module}."))
        else:
            diagnostics.append(_pass(f"import:{module}", f"Found {module}", str(spec.origin)))
    plugin_env = os.environ.get("MAPLE_CALCULATOR_PLUGINS")
    if plugin_env:
        diagnostics.append(
            _pass(
                "environment:MAPLE_CALCULATOR_PLUGINS",
                "The current shell exports MAPLE_CALCULATOR_PLUGINS, but Gau_MAPLE isolates calculator construction.",
                detail=(
                    f"value={plugin_env!r}; profile-specific plugins are loaded explicitly. "
                    "Raw MAPLE commands outside Gau_MAPLE may still inherit this shell value."
                ),
            )
        )
    else:
        diagnostics.append(
            _pass(
                "environment:MAPLE_CALCULATOR_PLUGINS",
                "No plugin contamination is present in the current shell.",
            )
        )
    return diagnostics


def _check_config(config: GauMapleConfig) -> list[Diagnostic]:
    diagnostics = [
        _pass("config", f"Loaded {config.source_path}"),
        _pass("runtime_dir", str(config.runtime_dir)),
    ]
    try:
        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        mode = oct(config.runtime_dir.stat().st_mode & 0o777)
        diagnostics.append(
            _pass("runtime_dir:writable", "Runtime directory is writable.", f"mode={mode}")
            if os.access(config.runtime_dir, os.W_OK | os.X_OK)
            else _fail("runtime_dir:writable", "Runtime directory is not writable.")
        )
    except OSError as exc:
        diagnostics.append(
            _fail("runtime_dir:writable", "Cannot create or inspect runtime directory.", str(exc))
        )
    return diagnostics


def _check_server_definition(
    config: GauMapleConfig,
    definition: ServerDefinition,
    *,
    probe_imports: bool,
    import_timeout: float,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    executable = definition.executable
    if executable.is_file() and os.access(executable, os.X_OK):
        diagnostics.append(
            _pass(f"server:{definition.name}:executable", str(executable))
        )
    else:
        diagnostics.append(
            _fail(
                f"server:{definition.name}:executable",
                "Configured server executable is missing or not executable.",
                str(executable),
            )
        )
        return diagnostics

    assigned = ", ".join(definition.profiles)
    diagnostics.append(
        _pass(f"server:{definition.name}:profiles", assigned)
    )

    # A paired Python interpreter is needed only for the optional import probe.
    # When import probing is disabled, validating the configured server
    # executable is sufficient.  This also permits wrapper executables and
    # minimal test fixtures such as /bin/true without inventing /bin/python.
    if probe_imports:
        python_executable = _python_for_server(definition)
        if python_executable.is_file() and os.access(python_executable, os.X_OK):
            diagnostics.append(
                _pass(f"server:{definition.name}:python", str(python_executable))
            )
        else:
            diagnostics.append(
                _fail(
                    f"server:{definition.name}:python",
                    "Cannot find the Python executable paired with this server.",
                    str(python_executable),
                )
            )
            return diagnostics

        modules = _profile_modules(config, definition)
        ok, detail = _run_import_probe(
            python_executable,
            modules,
            environment=_child_environment(definition),
            timeout=import_timeout,
        )
        if ok:
            diagnostics.append(
                _pass(
                    f"server:{definition.name}:imports",
                    "Environment imports succeeded.",
                    detail,
                )
            )
        else:
            diagnostics.append(
                _fail(
                    f"server:{definition.name}:imports",
                    "Environment import probe failed.",
                    detail,
                )
            )
    return diagnostics


def _check_profile_paths(config: GauMapleConfig) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for name, profile in config.profiles.items():
        raw = profile.model_options.get("model_path")
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file():
            diagnostics.append(_pass(f"profile:{name}:model_path", str(path)))
        else:
            diagnostics.append(
                _fail(
                    f"profile:{name}:model_path",
                    "Configured model file does not exist.",
                    str(path),
                )
            )
    return diagnostics


def _check_servers(config: GauMapleConfig, *, timeout: float) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for name, definition in config.servers.items():
        try:
            metadata = ping_server(
                definition.socket_path,
                timeout=timeout,
                expect_server=name,
            )
        except Exception as exc:
            diagnostics.append(
                _warn(
                    f"server:{name}:runtime",
                    "Server is not currently reachable.",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        failed_profiles = [
            profile_name
            for profile_name, status in metadata.profile_statuses.items()
            if status.get("preload_state") == "failed"
        ]
        if failed_profiles:
            diagnostics.append(
                _fail(
                    f"server:{name}:runtime",
                    "Server is reachable but one or more profiles failed to preload.",
                    ", ".join(failed_profiles),
                )
            )
        else:
            diagnostics.append(
                _pass(
                    f"server:{name}:runtime",
                    f"UP pid={metadata.pid} requests={metadata.request_count}",
                    ", ".join(metadata.available_profiles),
                )
            )
    return diagnostics


def _check_gaussian(path: Path | None) -> list[Diagnostic]:
    if path is None:
        return [_warn("gaussian", "Gaussian executable was not checked.")]
    resolved = path.expanduser().absolute()
    if resolved.is_file() and os.access(resolved, os.X_OK):
        return [_pass("gaussian", str(resolved))]
    return [_fail("gaussian", "Gaussian executable is missing or not executable.", str(resolved))]


def run_diagnostics(
    config: GauMapleConfig,
    *,
    gaussian: Path | None = None,
    probe_servers: bool = True,
    probe_imports: bool = True,
    timeout: float = 5.0,
    import_timeout: float = 30.0,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_check_python())
    diagnostics.extend(_check_current_imports())
    diagnostics.extend(_check_config(config))
    diagnostics.extend(_check_profile_paths(config))
    for definition in config.servers.values():
        diagnostics.extend(
            _check_server_definition(
                config,
                definition,
                probe_imports=probe_imports,
                import_timeout=import_timeout,
            )
        )
    if probe_servers:
        diagnostics.extend(_check_servers(config, timeout=timeout))
    diagnostics.extend(_check_gaussian(gaussian))
    return tuple(diagnostics)


def _print_human(diagnostics: Iterable[Diagnostic]) -> None:
    for item in diagnostics:
        print(f"{item.status:4s} {item.name}: {item.message}")
        if item.detail:
            print(f"     {item.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gau-maple-doctor", allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gaussian", type=Path)
    parser.add_argument("--skip-servers", action="store_true")
    parser.add_argument("--skip-import-probes", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--import-timeout", type=float, default=30.0)
    parser.add_argument("--strict", action="store_true", help="treat WARN as failure")
    parser.add_argument("--json", dest="json_path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = default_config_path(args.config)
        config = load_config(config_path)
    except ConfigError as exc:
        diagnostic = _fail("config", "Configuration could not be loaded.", str(exc))
        _print_human([diagnostic])
        return 2

    diagnostics = run_diagnostics(
        config,
        gaussian=args.gaussian,
        probe_servers=not args.skip_servers,
        probe_imports=not args.skip_import_probes,
        timeout=args.timeout,
        import_timeout=args.import_timeout,
    )
    _print_human(diagnostics)

    if args.json_path is not None:
        payload = {
            "gau_maple_version": __version__,
            "config": str(config.source_path),
            "diagnostics": [asdict(item) for item in diagnostics],
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_path}")

    failed = any(item.failed for item in diagnostics)
    warned = any(item.warned for item in diagnostics)
    return 2 if failed or (args.strict and warned) else 0


if __name__ == "__main__":
    raise SystemExit(main())

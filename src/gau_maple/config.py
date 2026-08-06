"""TOML configuration for Gau_MAPLE profiles and persistent servers.

User-editable model and server definitions live in one reproducible TOML file.
The parser is deliberately strict: unknown keys, ambiguous profile placement,
and unsafe relative runtime paths fail before a server is started.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from .errors import ConfigError
from .profiles import MapleProfile


_PROJECT_KEYS = {"runtime_dir"}
_PROFILE_KEYS = {
    "model",
    "device",
    "d4",
    "implicit",
    "solvent",
    "strict_charge_multiplicity",
    "charge_policy",
    "multiplicity_policy",
    "reject_mm_charges",
    "model_options",
    "solvation_options",
}
_SERVER_KEYS = {
    "executable",
    "profiles",
    "socket",
    "pid_file",
    "log",
    "stdout",
    "preload",
    "startup_timeout",
    "shutdown_timeout",
    "environment",
}


def _unknown_keys(raw: Mapping[str, Any], allowed: set[str], *, section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {section}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}."
        )


def _nonempty(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{field} must not be empty.")
    return text


def _bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false, got {value!r}.")
    return value


def _positive_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a positive number, got {value!r}.") from exc
    if result <= 0:
        raise ConfigError(f"{field} must be positive, got {result}.")
    return result


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a TOML table.")
    return MappingProxyType({str(k): v for k, v in value.items()})


def _string_mapping(value: Any, *, field: str) -> Mapping[str, str]:
    raw = _mapping(value, field=field)
    cleaned: dict[str, str] = {}
    for key, item in raw.items():
        name = _nonempty(key, field=f"{field} key")
        if not isinstance(item, (str, int, float, bool)):
            raise ConfigError(
                f"{field}.{name} must be a scalar convertible to an environment string."
            )
        if isinstance(item, bool):
            cleaned[name] = "1" if item else "0"
        else:
            cleaned[name] = str(item)
    return MappingProxyType(cleaned)


def _expand_template(
    value: Any,
    *,
    field: str,
    variables: Mapping[str, str],
) -> Path:
    text = _nonempty(value, field=field)
    try:
        expanded = text.format_map(variables)
    except KeyError as exc:
        allowed = ", ".join(sorted(variables))
        raise ConfigError(
            f"Unknown placeholder {{{exc.args[0]}}} in {field}; allowed: {allowed}."
        ) from exc
    path = Path(os.path.expandvars(os.path.expanduser(expanded)))
    if not path.is_absolute():
        raise ConfigError(f"{field} must resolve to an absolute path, got {path}.")
    return path


@dataclass(frozen=True, slots=True)
class ServerDefinition:
    """One persistent server process and the profiles assigned to it."""

    name: str
    executable: Path
    profiles: tuple[str, ...]
    socket_path: Path
    pid_file: Path
    log_path: Path
    stdout_path: Path
    preload: bool = True
    startup_timeout: float = 900.0
    shutdown_timeout: float = 20.0
    environment: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class GauMapleConfig:
    """Fully validated Gau_MAPLE configuration."""

    source_path: Path
    runtime_dir: Path
    profiles: Mapping[str, MapleProfile]
    servers: Mapping[str, ServerDefinition]
    profile_to_server: Mapping[str, str]

    def get_profile(self, name: str) -> MapleProfile:
        key = str(name).strip()
        try:
            return self.profiles[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles))
            raise ConfigError(
                f"Unknown profile {key!r}; available profiles: {available}."
            ) from exc

    def get_server(self, name: str) -> ServerDefinition:
        key = str(name).strip()
        try:
            return self.servers[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.servers))
            raise ConfigError(
                f"Unknown server {key!r}; available servers: {available}."
            ) from exc

    def server_for_profile(
        self,
        profile_name: str,
        *,
        server_name: str | None = None,
    ) -> ServerDefinition:
        profile = str(profile_name).strip()
        self.get_profile(profile)
        assigned = self.profile_to_server.get(profile)
        if assigned is None:
            raise ConfigError(f"Profile {profile!r} is not assigned to any server.")
        if server_name is not None and str(server_name).strip() != assigned:
            raise ConfigError(
                f"Profile {profile!r} belongs to server {assigned!r}, not "
                f"{str(server_name).strip()!r}."
            )
        return self.servers[assigned]

    def profiles_for_server(self, server_name: str) -> dict[str, MapleProfile]:
        server = self.get_server(server_name)
        return {name: self.profiles[name] for name in server.profiles}


def default_config_path(explicit: str | Path | None = None) -> Path:
    """Resolve a config path without depending on the caller's Conda environment."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_path = os.environ.get("GAU_MAPLE_CONFIG", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / "profiles.toml")
    for candidate in candidates:
        path = candidate.expanduser().absolute()
        if path.is_file():
            return path
    if explicit is not None:
        raise ConfigError(f"Gau_MAPLE config file does not exist: {Path(explicit).expanduser()}.")
    searched = ", ".join(str(p.expanduser().absolute()) for p in candidates)
    raise ConfigError(
        "No Gau_MAPLE config file was found. Pass --config PATH or set "
        f"GAU_MAPLE_CONFIG. Searched: {searched}."
    )


def load_config(path: str | Path) -> GauMapleConfig:
    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise ConfigError(f"Gau_MAPLE config file does not exist: {source}.")
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {source}: {exc}.") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Top-level TOML document must be a table.")
    top_unknown = sorted(set(raw) - {"project", "profiles", "servers"})
    if top_unknown:
        raise ConfigError(
            f"Unknown top-level section(s): {', '.join(top_unknown)}; "
            "expected [project], [profiles.*], and [servers.*]."
        )

    project = raw.get("project", {})
    if not isinstance(project, Mapping):
        raise ConfigError("[project] must be a TOML table.")
    _unknown_keys(project, _PROJECT_KEYS, section="[project]")

    variables = {
        "user": getpass.getuser(),
        "home": str(Path.home()),
        "project_dir": str(source.parent.parent),
    }
    runtime_dir = _expand_template(
        project.get("runtime_dir", "/tmp/gau-maple-{user}"),
        field="project.runtime_dir",
        variables=variables,
    )
    variables = dict(variables)
    variables["runtime_dir"] = str(runtime_dir)

    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ConfigError("At least one [profiles.NAME] table is required.")
    profiles: dict[str, MapleProfile] = {}
    for raw_name, value in raw_profiles.items():
        name = _nonempty(raw_name, field="profile name")
        if not isinstance(value, Mapping):
            raise ConfigError(f"[profiles.{name}] must be a TOML table.")
        _unknown_keys(value, _PROFILE_KEYS, section=f"[profiles.{name}]")
        if "model" not in value:
            raise ConfigError(f"[profiles.{name}] is missing required key 'model'.")
        profiles[name] = MapleProfile(
            name=name,
            model=_nonempty(value["model"], field=f"profiles.{name}.model"),
            device=_nonempty(value.get("device", "cpu"), field=f"profiles.{name}.device"),
            model_options=_mapping(
                value.get("model_options"), field=f"profiles.{name}.model_options"
            ),
            d4=_bool(value.get("d4", False), field=f"profiles.{name}.d4"),
            implicit=_nonempty(
                value.get("implicit", "none"), field=f"profiles.{name}.implicit"
            ),
            solvent=_nonempty(
                value.get("solvent", "none"), field=f"profiles.{name}.solvent"
            ),
            solvation_options=_mapping(
                value.get("solvation_options"),
                field=f"profiles.{name}.solvation_options",
            ),
            strict_charge_multiplicity=_bool(
                value.get("strict_charge_multiplicity", True),
                field=f"profiles.{name}.strict_charge_multiplicity",
            ),
            charge_policy=_nonempty(
                value.get("charge_policy", "calculator"),
                field=f"profiles.{name}.charge_policy",
            ),
            multiplicity_policy=_nonempty(
                value.get("multiplicity_policy", "calculator"),
                field=f"profiles.{name}.multiplicity_policy",
            ),
            reject_mm_charges=_bool(
                value.get("reject_mm_charges", True),
                field=f"profiles.{name}.reject_mm_charges",
            ),
        )

    raw_servers = raw.get("servers")
    if not isinstance(raw_servers, Mapping) or not raw_servers:
        raise ConfigError("At least one [servers.NAME] table is required.")
    servers: dict[str, ServerDefinition] = {}
    profile_to_server: dict[str, str] = {}
    for raw_name, value in raw_servers.items():
        name = _nonempty(raw_name, field="server name")
        if not isinstance(value, Mapping):
            raise ConfigError(f"[servers.{name}] must be a TOML table.")
        _unknown_keys(value, _SERVER_KEYS, section=f"[servers.{name}]")
        if "executable" not in value:
            raise ConfigError(f"[servers.{name}] is missing required key 'executable'.")
        raw_assigned = value.get("profiles")
        if not isinstance(raw_assigned, list) or not raw_assigned:
            raise ConfigError(
                f"servers.{name}.profiles must be a non-empty TOML array of profile names."
            )
        assigned: list[str] = []
        for item in raw_assigned:
            profile_name = _nonempty(item, field=f"servers.{name}.profiles item")
            if profile_name not in profiles:
                raise ConfigError(
                    f"Server {name!r} references unknown profile {profile_name!r}."
                )
            if profile_name in assigned:
                raise ConfigError(
                    f"Server {name!r} lists profile {profile_name!r} more than once."
                )
            previous = profile_to_server.get(profile_name)
            if previous is not None:
                raise ConfigError(
                    f"Profile {profile_name!r} is assigned to both {previous!r} and "
                    f"{name!r}; profile-to-server routing must be unique."
                )
            assigned.append(profile_name)
            profile_to_server[profile_name] = name

        socket_path = _expand_template(
            value.get("socket", "{runtime_dir}/" + name + ".sock"),
            field=f"servers.{name}.socket",
            variables=variables,
        )
        pid_file = _expand_template(
            value.get("pid_file", "{runtime_dir}/" + name + ".pid"),
            field=f"servers.{name}.pid_file",
            variables=variables,
        )
        log_path = _expand_template(
            value.get("log", "{runtime_dir}/" + name + ".server.log"),
            field=f"servers.{name}.log",
            variables=variables,
        )
        stdout_path = _expand_template(
            value.get("stdout", "{runtime_dir}/" + name + ".stdout.log"),
            field=f"servers.{name}.stdout",
            variables=variables,
        )
        executable = _expand_template(
            value["executable"],
            field=f"servers.{name}.executable",
            variables=variables,
        )
        servers[name] = ServerDefinition(
            name=name,
            executable=executable,
            profiles=tuple(assigned),
            socket_path=socket_path,
            pid_file=pid_file,
            log_path=log_path,
            stdout_path=stdout_path,
            preload=_bool(value.get("preload", True), field=f"servers.{name}.preload"),
            startup_timeout=_positive_float(
                value.get("startup_timeout", 900.0),
                field=f"servers.{name}.startup_timeout",
            ),
            shutdown_timeout=_positive_float(
                value.get("shutdown_timeout", 20.0),
                field=f"servers.{name}.shutdown_timeout",
            ),
            environment=_string_mapping(
                value.get("environment"), field=f"servers.{name}.environment"
            ),
        )

    unassigned = sorted(set(profiles) - set(profile_to_server))
    if unassigned:
        raise ConfigError(
            "Every profile must belong to exactly one server; unassigned profiles: "
            + ", ".join(unassigned)
            + "."
        )

    return GauMapleConfig(
        source_path=source,
        runtime_dir=runtime_dir,
        profiles=MappingProxyType(profiles),
        servers=MappingProxyType(servers),
        profile_to_server=MappingProxyType(profile_to_server),
    )


def validate_runtime(config: GauMapleConfig) -> list[str]:
    """Return runtime diagnostics; raise on missing/non-executable server binaries."""

    diagnostics: list[str] = []
    for name, server in config.servers.items():
        if not server.executable.is_file():
            raise ConfigError(
                f"Server executable for {name!r} does not exist: {server.executable}."
            )
        if not os.access(server.executable, os.X_OK):
            raise ConfigError(
                f"Server executable for {name!r} is not executable: {server.executable}."
            )
        diagnostics.append(
            f"{name}: executable={server.executable} socket={server.socket_path} "
            f"profiles={','.join(server.profiles)} preload={server.preload}"
        )
    return diagnostics

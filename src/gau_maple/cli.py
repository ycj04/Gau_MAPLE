"""Command-line parsing for Gau_MAPLE direct and socket modes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .config import default_config_path, load_config
from .errors import ConfigError, ProfileError
from .profiles import MapleProfile

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?$"
)


def parse_scalar(text: str) -> Any:
    """Parse a conservative CLI scalar while preserving ordinary strings."""
    raw = str(text).strip()
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    if _INT_RE.fullmatch(raw):
        try:
            return int(raw)
        except ValueError:
            pass
    if _FLOAT_RE.fullmatch(raw):
        try:
            return float(raw.replace("D", "E").replace("d", "e"))
        except ValueError:
            pass
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        try:
            value = json.loads(raw)
            if isinstance(value, str):
                return value
        except json.JSONDecodeError:
            pass
    return raw


def parse_key_value(values: Sequence[str] | None, *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise ProfileError(f"{label} must use KEY=VALUE syntax, got {item!r}.")
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ProfileError(f"{label} contains an empty key in {item!r}.")
        if key in result:
            raise ProfileError(f"Duplicate {label} key: {key!r}.")
        result[key] = parse_scalar(raw_value)
    return result


def add_profile_arguments(
    parser: argparse.ArgumentParser,
    *,
    model_required: bool,
) -> None:
    parser.add_argument(
        "--model",
        required=model_required,
        help="MAPLE registry model name",
    )
    parser.add_argument("--device", default="cpu", help="MAPLE device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--profile-name",
        default="direct" if not model_required else "server",
        help="diagnostic name for this model configuration",
    )
    parser.add_argument(
        "--module",
        help="optional MAPLE calculator plugin module imported through model_options",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable MAPLE model option",
    )
    parser.add_argument("--d4", action="store_true", help="request MAPLE D4 when supported")
    parser.add_argument("--implicit", default="none")
    parser.add_argument("--solvent", default="none")
    parser.add_argument(
        "--solvation-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--allow-unsupported-charge-mult",
        action="store_true",
        help=(
            "permit a charged/open-shell request even when a calculator declares "
            "SUPPORTS_CHARGE_MULT=False; use only for deliberate diagnostics"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="also print a Python traceback to stderr on failure",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gau-maple",
        description=(
            "Gaussian External adapter. Use --model for direct mode or --socket "
            "to connect to a persistent local Gau_MAPLE server."
        ),
        allow_abbrev=False,
    )
    add_profile_arguments(parser, model_required=False)
    parser.add_argument(
        "--backend-log",
        type=Path,
        help="optional separate MAPLE backend log in direct mode",
    )
    parser.add_argument(
        "--socket",
        type=Path,
        help="connect to a persistent local Unix domain socket server",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="socket request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--profile",
        help="profile to request from a multi-profile socket server",
    )
    parser.add_argument(
        "--expect-server",
        help="reject a socket server whose server identity differs",
    )
    parser.add_argument(
        "--expect-profile",
        help=(
            "backward-compatible alias/check for --profile; when --profile is "
            "omitted this value also selects the remote profile"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "TOML config. With --profile and no --socket, resolve the "
            "correct persistent server automatically."
        ),
    )
    parser.add_argument(
        "--server",
        help="optional server identity check/disambiguation in config mode",
    )
    return parser


def profile_from_namespace(args: argparse.Namespace) -> MapleProfile:
    if not getattr(args, "model", None):
        raise ProfileError(
            "Direct/server model construction requires --model. In client socket "
            "mode, omit profile construction and use --socket instead."
        )
    model_options = parse_key_value(args.option, label="--option")
    if args.module:
        if "module" in model_options and model_options["module"] != args.module:
            raise ProfileError(
                "Plugin module was specified both through --module and --option module=... "
                "with different values."
            )
        model_options["module"] = str(args.module).strip()
    solvation_options = parse_key_value(
        args.solvation_option,
        label="--solvation-option",
    )
    return MapleProfile(
        name=args.profile_name,
        model=args.model,
        device=args.device,
        model_options=model_options,
        d4=args.d4,
        implicit=args.implicit,
        solvent=args.solvent,
        solvation_options=solvation_options,
        strict_charge_multiplicity=not args.allow_unsupported_charge_mult,
        reject_mm_charges=True,
    )


def remote_profile_from_namespace(args: argparse.Namespace) -> str | None:
    profile = getattr(args, "profile", None)
    expected = getattr(args, "expect_profile", None)
    if profile and expected and profile != expected:
        raise ProfileError(
            f"--profile={profile!r} and --expect-profile={expected!r} disagree."
        )
    selected = profile or expected
    return None if selected is None else str(selected).strip()



def resolve_config_client(args: argparse.Namespace) -> None:
    """Resolve TOML profile routing into socket and expected server in-place.

    Explicit ``--config``/``--server`` always selects config mode.  A bare
    ``--profile`` also selects it when neither direct ``--model`` nor explicit
    ``--socket`` was provided; the config then comes from GAU_MAPLE_CONFIG or
    the documented local default path.
    """

    selected = remote_profile_from_namespace(args)
    config_intent = (
        args.config is not None
        or args.server is not None
        or (args.socket is None and args.model is None and selected is not None)
    )
    if not config_intent:
        return
    if args.socket is not None:
        raise ConfigError("Use either --config or --socket, not both.")
    if not selected:
        raise ConfigError("Config client mode requires --profile PROFILE.")
    config_path = default_config_path(args.config)
    config = load_config(config_path)
    definition = config.server_for_profile(selected, server_name=args.server)
    args.socket = definition.socket_path
    if args.expect_server and args.expect_server != definition.name:
        raise ConfigError(
            f"--expect-server={args.expect_server!r} disagrees with TOML routing "
            f"to {definition.name!r}."
        )
    args.expect_server = definition.name


def validate_external_mode(args: argparse.Namespace) -> str:
    """Return 'socket' or 'direct' and reject ambiguous CLI combinations."""
    if args.socket:
        profile_fields = {
            "--model": args.model,
            "--module": args.module,
            "--option": args.option,
            "--d4": args.d4,
            "--implicit": None if args.implicit == "none" else args.implicit,
            "--solvent": None if args.solvent == "none" else args.solvent,
            "--solvation-option": args.solvation_option,
            "--allow-unsupported-charge-mult": args.allow_unsupported_charge_mult,
            "--backend-log": args.backend_log,
        }
        supplied = [key for key, value in profile_fields.items() if value]
        if supplied:
            raise ProfileError(
                "Socket mode uses profiles loaded by the server; remove "
                f"direct-mode option(s): {', '.join(supplied)}."
            )
        if args.timeout <= 0:
            raise ProfileError("--timeout must be positive.")
        selected = remote_profile_from_namespace(args)
        if not selected:
            raise ProfileError(
                "Multi-profile socket mode requires --profile PROFILE "
                "(or the legacy --expect-profile alias)."
            )
        return "socket"
    if not args.model:
        raise ProfileError(
            "Choose direct mode with --model, explicit socket mode with --socket, "
            "or config mode with --config/--profile."
        )
    if args.profile or args.expect_profile or args.expect_server or args.config or args.server:
        raise ProfileError(
            "--profile/--expect-profile/--expect-server/--config/--server are "
            "valid only for persistent server mode."
        )
    return "direct"

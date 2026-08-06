"""Optional built-in profile groups for two common MAPLE runtimes.

Production installations should prefer the user-editable TOML configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
import os

from .errors import ProfileError
from .profiles import MapleProfile

ESEN_MODEL_PATH = os.environ.get(
    "GAU_MAPLE_ESEN_MODEL_PATH", "esen_sm_conserving_all.pt"
)


def maple_server_profiles() -> dict[str, MapleProfile]:
    """Profiles intended for the ``maple`` Conda environment.

    A profile being listed does not replace its own smoke test; eager preload
    records any backend that fails without killing the remaining server.
    """

    return {
        "aimnet2": MapleProfile(
            name="aimnet2",
            model="aimnet2",
            device="cpu",
            model_options={"hessian": "analytic"},
        ),
        "aimnet2nse": MapleProfile(
            name="aimnet2nse",
            model="aimnet2nse",
            device="cpu",
            model_options={"hessian": "analytic"},
        ),
        "ani1x": MapleProfile(name="ani1x", model="ani1x", device="cpu"),
        "ani1ccx": MapleProfile(name="ani1ccx", model="ani1ccx", device="cpu"),
        "ani1xnr": MapleProfile(name="ani1xnr", model="ani1xnr", device="cpu"),
        "ani2x": MapleProfile(name="ani2x", model="ani2x", device="cpu"),
        "maceoff23m": MapleProfile(
            name="maceoff23m",
            model="maceoff23m",
            device="cpu",
        ),
        "maceomol_native": MapleProfile(
            name="maceomol_native",
            model="maceomol_native",
            device="cpu",
            model_options={"module": "maple_mace_native"},
        ),
        "macepolm_native": MapleProfile(
            name="macepolm_native",
            model="macepolm_native",
            device="cpu",
            model_options={"module": "maple_mace_native"},
        ),
    }


def meta_server_profiles() -> dict[str, MapleProfile]:
    """Profiles intended for the ``uma_env`` Conda environment."""

    return {
        "uma-s-1p2": MapleProfile(
            name="uma-s-1p2",
            model="uma",
            device="cpu",
            model_options={
                "size": "uma-s-1p2",
                "task": "omol",
                "inference": "default",
            },
        ),
        "esen-sm-conserving-all": MapleProfile(
            name="esen-sm-conserving-all",
            model="esen_sm_conserving_all",
            device="cpu",
            model_options={
                "module": "maple.function.calculator.esen_plugin",
                "model_path": ESEN_MODEL_PATH,
            },
        ),
    }


def get_server_group(name: str) -> tuple[str, dict[str, MapleProfile]]:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"maple", "maple_server"}:
        return "maple_server", maple_server_profiles()
    if normalized in {"meta", "meta_server"}:
        return "meta_server", meta_server_profiles()
    raise ProfileError(
        f"Unknown server group {name!r}; expected 'maple' or 'meta'."
    )


def validate_profile_mapping(
    profiles: Mapping[str, MapleProfile],
) -> dict[str, MapleProfile]:
    cleaned: dict[str, MapleProfile] = {}
    for raw_name, profile in profiles.items():
        name = str(raw_name).strip()
        if not name:
            raise ProfileError("Server profile names must not be empty.")
        if name in cleaned:
            raise ProfileError(f"Duplicate server profile name: {name!r}.")
        if profile.name != name:
            raise ProfileError(
                f"Profile mapping key {name!r} does not match profile.name "
                f"{profile.name!r}."
            )
        cleaned[name] = profile
    if not cleaned:
        raise ProfileError("A Gau_MAPLE server must contain at least one profile.")
    return cleaned

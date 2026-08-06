"""Typed configuration for MAPLE calculator backends.

Configuration parsing is intentionally kept separate from profile definitions.  This
load these profiles from TOML; for now callers construct :class:`MapleProfile`
directly so backend behaviour can be tested without a CLI or server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ProfileError


def _clean_name(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ProfileError(f"{field_name} must not be empty.")
    return text


def _clean_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProfileError(f"{field_name} must be a mapping, got {type(value).__name__}.")
    cleaned: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ProfileError(f"{field_name} contains an empty key.")
        cleaned[key] = raw_value
    return MappingProxyType(cleaned)


@dataclass(frozen=True, slots=True)
class MapleProfile:
    """Construction options for one cached MAPLE calculator.

    ``model_options`` is passed to MAPLE's ``SetCalculator`` unchanged except
    for making a defensive copy.  External plugins should be selected with the
    current MAPLE convention, for example::

        MapleProfile(
            name="maceomol_native",
            model="maceomol_native",
            model_options={"module": "maple_mace_native"},
        )

    Strict flags are Gau_MAPLE policies rather than MAPLE model options.
    """

    name: str
    model: str
    device: str = "cpu"
    model_options: Mapping[str, Any] = field(default_factory=dict)
    d4: bool = False
    implicit: str = "none"
    solvent: str = "none"
    solvation_options: Mapping[str, Any] = field(default_factory=dict)
    strict_charge_multiplicity: bool = True
    charge_policy: str = "calculator"
    multiplicity_policy: str = "calculator"
    reject_mm_charges: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field_name="name"))
        object.__setattr__(self, "model", _clean_name(self.model, field_name="model").lower())
        object.__setattr__(self, "device", _clean_name(self.device, field_name="device"))
        object.__setattr__(self, "implicit", _clean_name(self.implicit, field_name="implicit").lower())
        object.__setattr__(self, "solvent", _clean_name(self.solvent, field_name="solvent").lower())
        object.__setattr__(
            self,
            "model_options",
            _clean_mapping(self.model_options, field_name="model_options"),
        )
        object.__setattr__(
            self,
            "solvation_options",
            _clean_mapping(self.solvation_options, field_name="solvation_options"),
        )
        charge_policy = _clean_name(
            self.charge_policy, field_name="charge_policy"
        ).lower()
        multiplicity_policy = _clean_name(
            self.multiplicity_policy, field_name="multiplicity_policy"
        ).lower()
        allowed_charge = {"calculator", "supported", "neutral_only"}
        allowed_multiplicity = {"calculator", "supported", "singlet_only"}
        if charge_policy not in allowed_charge:
            raise ProfileError(
                "charge_policy must be one of "
                f"{sorted(allowed_charge)}, got {charge_policy!r}."
            )
        if multiplicity_policy not in allowed_multiplicity:
            raise ProfileError(
                "multiplicity_policy must be one of "
                f"{sorted(allowed_multiplicity)}, got {multiplicity_policy!r}."
            )
        object.__setattr__(self, "charge_policy", charge_policy)
        object.__setattr__(self, "multiplicity_policy", multiplicity_policy)

    def factory_kwargs(self) -> dict[str, Any]:
        """Return exactly the keyword arguments expected by SetCalculator."""
        return {
            "device": self.device,
            "model": self.model,
            "d4": bool(self.d4),
            "implicit": self.implicit,
            "solvent": self.solvent,
            "model_options": dict(self.model_options),
            "solvation_options": dict(self.solvation_options),
        }

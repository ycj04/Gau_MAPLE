"""Cross-profile capability and persistent-server stability checks.

These checks complement Gaussian input workflows:

* capability probes verify that charge/multiplicity requests are either accepted
  or rejected according to the configured engineering contract;
* stability probes repeat and interleave server requests to detect stale ASE
  calculator state, profile routing mistakes, and atom-count contamination.

A successful charge/spin probe demonstrates only interface propagation and
finite output.  It does not establish chemical accuracy for an electronic
state that may be outside a checkpoint's training domain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .client import evaluate_via_server
from .config import GauMapleConfig, default_config_path, load_config
from .errors import RemoteServerError
from .models import ExternalRequest, ExternalResult
from .units import BOHR_TO_ANGSTROM


PROFILE_ELECTRONIC_STATE_CAPABILITIES: dict[str, tuple[bool, bool]] = {
    # profile: (supports_charge, supports_multiplicity)
    "aimnet2": (True, False),
    "aimnet2nse": (True, True),
    "ani1x": (False, False),
    "ani1ccx": (False, False),
    "ani1xnr": (False, False),
    "ani2x": (False, False),
    "maceoff23m": (False, False),
    "maceomol_native": (True, True),
    "macepolm_native": (True, True),
    "uma-s-1p2": (True, True),
    "esen-sm-conserving-all": (True, True),
}

CHARGE_SUPPORTED_PROFILES = tuple(
    profile
    for profile, (supports_charge, _supports_multiplicity)
    in PROFILE_ELECTRONIC_STATE_CAPABILITIES.items()
    if supports_charge
)
MULTIPLICITY_SUPPORTED_PROFILES = tuple(
    profile
    for profile, (_supports_charge, supports_multiplicity)
    in PROFILE_ELECTRONIC_STATE_CAPABILITIES.items()
    if supports_multiplicity
)


DEFAULT_STABILITY_PROFILES = (
    "aimnet2",
    "aimnet2nse",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "ani2x",
    "maceoff23m",
    "maceomol_native",
    "macepolm_native",
    "uma-s-1p2",
    "esen-sm-conserving-all",
)

# MAPLE/torch float32 backends can be deterministic to different practical
# tolerances. Keep the default server strict, but allow the observed, harmless
# 1e-8 Ha-scale reduction noise in the uma_env backends.
DEFAULT_ENERGY_ATOL = 1.0e-9
DEFAULT_GRADIENT_ATOL = 1.0e-8
PROFILE_STABILITY_ATOL: dict[str, tuple[float, float]] = {
    "uma-s-1p2": (1.0e-7, 1.0e-7),
    "esen-sm-conserving-all": (1.0e-7, 1.0e-7),
}


def _stability_tolerances(
    profile: str,
    *,
    energy_atol: float | None,
    gradient_atol: float | None,
) -> tuple[float, float]:
    recommended_energy, recommended_gradient = PROFILE_STABILITY_ATOL.get(
        profile,
        (DEFAULT_ENERGY_ATOL, DEFAULT_GRADIENT_ATOL),
    )
    return (
        recommended_energy if energy_atol is None else float(energy_atol),
        recommended_gradient if gradient_atol is None else float(gradient_atol),
    )


def _request(
    numbers: Sequence[int],
    positions_angstrom: Sequence[Sequence[float]],
    *,
    charge: int = 0,
    multiplicity: int = 1,
    derivative_order: int = 1,
) -> ExternalRequest:
    positions = np.asarray(positions_angstrom, dtype=np.float64) / BOHR_TO_ANGSTROM
    numbers_array = np.asarray(numbers, dtype=np.int64)
    return ExternalRequest(
        atomic_numbers=numbers_array,
        positions_bohr=positions,
        derivative_order=derivative_order,
        charge=charge,
        multiplicity=multiplicity,
        mm_charges=np.zeros(len(numbers_array), dtype=np.float64),
    )


def water_request(*, derivative_order: int = 1) -> ExternalRequest:
    return _request(
        [8, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [0.75716, 0.0, 0.58626],
            [-0.75716, 0.0, 0.58626],
        ],
        derivative_order=derivative_order,
    )


def ammonia_request(*, derivative_order: int = 1) -> ExternalRequest:
    return _request(
        [7, 1, 1, 1],
        [
            [0.0, 0.0, 0.116],
            [0.9377, 0.0, -0.271],
            [-0.46885, 0.81208, -0.271],
            [-0.46885, -0.81208, -0.271],
        ],
        derivative_order=derivative_order,
    )


def ammonium_request() -> ExternalRequest:
    return _request(
        [7, 1, 1, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [0.59467, 0.59467, 0.59467],
            [-0.59467, -0.59467, 0.59467],
            [-0.59467, 0.59467, -0.59467],
            [0.59467, -0.59467, -0.59467],
        ],
        charge=1,
        multiplicity=1,
        derivative_order=0,
    )


def hydroxyl_request() -> ExternalRequest:
    return _request(
        [8, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
        charge=0,
        multiplicity=2,
        derivative_order=0,
    )


def water_cation_request() -> ExternalRequest:
    return _request(
        [8, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [0.75716, 0.0, 0.58626],
            [-0.75716, 0.0, 0.58626],
        ],
        charge=1,
        multiplicity=2,
        derivative_order=0,
    )


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    profile: str
    case: str
    expected: str
    observed: str
    passed: bool
    energy_hartree: float | None = None
    error: str | None = None




@dataclass(frozen=True, slots=True)
class InterleaveProbe:
    server: str
    profiles: tuple[str, ...]
    cycles: int
    passed: bool
    energy_atol_hartree: float
    gradient_atol_hartree_per_bohr: float
    max_energy_deviation_hartree: float | None
    max_gradient_deviation_hartree_per_bohr: float | None
    error: str | None = None

@dataclass(frozen=True, slots=True)
class StabilityProbe:
    profile: str
    repeats: int
    passed: bool
    energy_atol_hartree: float
    gradient_atol_hartree_per_bohr: float
    energy_span_hartree: float | None
    max_gradient_span_hartree_per_bohr: float | None
    atom_count_roundtrip_passed: bool
    error: str | None = None


def _evaluate(
    config: GauMapleConfig,
    profile: str,
    request: ExternalRequest,
    *,
    timeout: float,
) -> ExternalResult:
    server = config.server_for_profile(profile)
    result, _metadata = evaluate_via_server(
        request,
        server.socket_path,
        profile_name=profile,
        timeout=timeout,
        expect_server=server.name,
        expect_profile=profile,
    )
    return result


def _is_capability_rejection(message: str) -> bool:
    lowered = message.lower()
    return (
        "supports_charge_mult=false" in lowered
        or "charge_policy='neutral_only'" in lowered
        or "multiplicity_policy='singlet_only'" in lowered
        or "charge/multiplicity" in lowered
        or ("charge=" in lowered and "multiplicity=" in lowered)
    )


def run_capability_probes(
    config: GauMapleConfig,
    *,
    timeout: float = 600.0,
    evaluator: Callable[[GauMapleConfig, str, ExternalRequest], ExternalResult] | None = None,
) -> tuple[CapabilityProbe, ...]:
    """Probe charge-only, multiplicity-only, and combined-state routing."""

    if evaluator is None:
        def evaluator(cfg: GauMapleConfig, profile: str, request: ExternalRequest) -> ExternalResult:
            return _evaluate(cfg, profile, request, timeout=timeout)

    cases = (
        ("charge_closed_shell", ammonium_request(), "charge"),
        ("open_shell_doublet", hydroxyl_request(), "multiplicity"),
        ("charged_open_shell", water_cation_request(), "combined"),
    )
    probes: list[CapabilityProbe] = []
    for profile, (supports_charge, supports_multiplicity) in (
        PROFILE_ELECTRONIC_STATE_CAPABILITIES.items()
    ):
        for case_name, request, dimension in cases:
            if dimension == "charge":
                expected = "accept" if supports_charge else "reject"
            elif dimension == "multiplicity":
                expected = "accept" if supports_multiplicity else "reject"
            else:
                expected = (
                    "accept"
                    if supports_charge and supports_multiplicity
                    else "reject"
                )
            try:
                result = evaluator(config, profile, request)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                observed = "reject" if _is_capability_rejection(message) else "error"
                probes.append(
                    CapabilityProbe(
                        profile=profile,
                        case=case_name,
                        expected=expected,
                        observed=observed,
                        passed=(expected == "reject" and observed == "reject"),
                        error=message,
                    )
                )
            else:
                probes.append(
                    CapabilityProbe(
                        profile=profile,
                        case=case_name,
                        expected=expected,
                        observed="accept",
                        passed=(expected == "accept"),
                        energy_hartree=float(result.energy_hartree),
                    )
                )
    return tuple(probes)


def _gradient_array(result: ExternalResult) -> np.ndarray:
    value = result.gradient_hartree_per_bohr
    if value is None:
        raise ValueError("stability request returned no gradient")
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("stability request returned a non-finite gradient")
    return array


def run_stability_probes(
    config: GauMapleConfig,
    *,
    profiles: Iterable[str] = DEFAULT_STABILITY_PROFILES,
    repeats: int = 5,
    timeout: float = 600.0,
    energy_atol: float | None = None,
    gradient_atol: float | None = None,
    evaluator: Callable[[GauMapleConfig, str, ExternalRequest], ExternalResult] | None = None,
) -> tuple[StabilityProbe, ...]:
    """Repeat and interleave requests to expose mutable-calculator state leaks."""

    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    if evaluator is None:
        def evaluator(cfg: GauMapleConfig, profile: str, request: ExternalRequest) -> ExternalResult:
            return _evaluate(cfg, profile, request, timeout=timeout)

    probes: list[StabilityProbe] = []
    water = water_request()
    ammonia = ammonia_request()
    for profile in tuple(profiles):
        profile_energy_atol, profile_gradient_atol = _stability_tolerances(
            profile,
            energy_atol=energy_atol,
            gradient_atol=gradient_atol,
        )
        try:
            results = [evaluator(config, profile, water) for _ in range(repeats)]
            energies = np.asarray([item.energy_hartree for item in results], dtype=np.float64)
            gradients = np.stack([_gradient_array(item) for item in results])
            energy_span = float(np.ptp(energies))
            gradient_span = float(np.max(np.ptp(gradients, axis=0)))

            first = results[0]
            evaluator(config, profile, ammonia)
            roundtrip = evaluator(config, profile, water)
            atom_count_ok = bool(
                abs(roundtrip.energy_hartree - first.energy_hartree)
                <= profile_energy_atol
                and float(
                    np.max(
                        np.abs(
                            _gradient_array(roundtrip)
                            - _gradient_array(first)
                        )
                    )
                )
                <= profile_gradient_atol
            )
            passed = bool(
                energy_span <= profile_energy_atol
                and gradient_span <= profile_gradient_atol
                and atom_count_ok
            )
            probes.append(
                StabilityProbe(
                    profile=profile,
                    repeats=repeats,
                    passed=passed,
                    energy_atol_hartree=profile_energy_atol,
                    gradient_atol_hartree_per_bohr=profile_gradient_atol,
                    energy_span_hartree=energy_span,
                    max_gradient_span_hartree_per_bohr=gradient_span,
                    atom_count_roundtrip_passed=atom_count_ok,
                )
            )
        except Exception as exc:
            probes.append(
                StabilityProbe(
                    profile=profile,
                    repeats=repeats,
                    passed=False,
                    energy_atol_hartree=profile_energy_atol,
                    gradient_atol_hartree_per_bohr=profile_gradient_atol,
                    energy_span_hartree=None,
                    max_gradient_span_hartree_per_bohr=None,
                    atom_count_roundtrip_passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(probes)



def run_interleave_probes(
    config: GauMapleConfig,
    *,
    profiles: Iterable[str] = DEFAULT_STABILITY_PROFILES,
    cycles: int = 3,
    timeout: float = 600.0,
    energy_atol: float | None = None,
    gradient_atol: float | None = None,
    evaluator: Callable[[GauMapleConfig, str, ExternalRequest], ExternalResult] | None = None,
) -> tuple[InterleaveProbe, ...]:
    """Alternate profiles within each server and compare against references."""

    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if evaluator is None:
        def evaluator(cfg: GauMapleConfig, profile: str, request: ExternalRequest) -> ExternalResult:
            return _evaluate(cfg, profile, request, timeout=timeout)

    grouped: dict[str, list[str]] = {}
    for profile in tuple(profiles):
        server = config.server_for_profile(profile)
        grouped.setdefault(server.name, []).append(profile)

    request = water_request()
    probes: list[InterleaveProbe] = []
    for server_name, server_profiles in grouped.items():
        profile_tolerances = [
            _stability_tolerances(
                profile,
                energy_atol=energy_atol,
                gradient_atol=gradient_atol,
            )
            for profile in server_profiles
        ]
        server_energy_atol = max(item[0] for item in profile_tolerances)
        server_gradient_atol = max(item[1] for item in profile_tolerances)
        try:
            references = {
                profile: evaluator(config, profile, request)
                for profile in server_profiles
            }
            max_energy = 0.0
            max_gradient = 0.0
            for _cycle in range(cycles):
                for profile in server_profiles:
                    result = evaluator(config, profile, request)
                    reference = references[profile]
                    max_energy = max(
                        max_energy,
                        abs(result.energy_hartree - reference.energy_hartree),
                    )
                    max_gradient = max(
                        max_gradient,
                        float(np.max(np.abs(
                            _gradient_array(result) - _gradient_array(reference)
                        ))),
                    )
            probes.append(
                InterleaveProbe(
                    server=server_name,
                    profiles=tuple(server_profiles),
                    cycles=cycles,
                    passed=bool(
                        max_energy <= server_energy_atol
                        and max_gradient <= server_gradient_atol
                    ),
                    energy_atol_hartree=server_energy_atol,
                    gradient_atol_hartree_per_bohr=server_gradient_atol,
                    max_energy_deviation_hartree=max_energy,
                    max_gradient_deviation_hartree_per_bohr=max_gradient,
                )
            )
        except Exception as exc:
            probes.append(
                InterleaveProbe(
                    server=server_name,
                    profiles=tuple(server_profiles),
                    cycles=cycles,
                    passed=False,
                    energy_atol_hartree=server_energy_atol,
                    gradient_atol_hartree_per_bohr=server_gradient_atol,
                    max_energy_deviation_hartree=None,
                    max_gradient_deviation_hartree_per_bohr=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(probes)

def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    print(f"JSON report: {path}")


def _print_capabilities(probes: Sequence[CapabilityProbe]) -> None:
    for probe in probes:
        status = "PASS" if probe.passed else "FAIL"
        detail = (
            f"{status} profile={probe.profile} case={probe.case} "
            f"expected={probe.expected} observed={probe.observed}"
        )
        if probe.energy_hartree is not None:
            detail += f" energy_hartree={probe.energy_hartree:.12f}"
        if probe.error:
            detail += f" error={probe.error}"
        print(detail)


def _print_stability(probes: Sequence[StabilityProbe]) -> None:
    for probe in probes:
        status = "PASS" if probe.passed else "FAIL"
        energy_text = (
            "-" if probe.energy_span_hartree is None
            else f"{probe.energy_span_hartree:.3e}"
        )
        gradient_text = (
            "-" if probe.max_gradient_span_hartree_per_bohr is None
            else f"{probe.max_gradient_span_hartree_per_bohr:.3e}"
        )
        print(
            f"{status} profile={probe.profile} repeats={probe.repeats} "
            f"energy_span={energy_text} "
            f"gradient_span={gradient_text} "
            f"energy_atol={probe.energy_atol_hartree:.1e} "
            f"gradient_atol={probe.gradient_atol_hartree_per_bohr:.1e} "
            f"atom_count_roundtrip={probe.atom_count_roundtrip_passed} "
            f"error={probe.error or '-'}"
        )



def _print_interleaving(probes: Sequence[InterleaveProbe]) -> None:
    for probe in probes:
        status = "PASS" if probe.passed else "FAIL"
        energy_text = (
            "-" if probe.max_energy_deviation_hartree is None
            else f"{probe.max_energy_deviation_hartree:.3e}"
        )
        gradient_text = (
            "-" if probe.max_gradient_deviation_hartree_per_bohr is None
            else f"{probe.max_gradient_deviation_hartree_per_bohr:.3e}"
        )
        print(
            f"{status} server={probe.server} cycles={probe.cycles} "
            f"profiles={','.join(probe.profiles)} "
            f"max_energy_deviation={energy_text} "
            f"max_gradient_deviation={gradient_text} "
            f"energy_atol={probe.energy_atol_hartree:.1e} "
            f"gradient_atol={probe.gradient_atol_hartree_per_bohr:.1e} "
            f"error={probe.error or '-'}"
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gau-maple-validation", allow_abbrev=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML configuration path; defaults to GAU_MAPLE_CONFIG or ./config/profiles.toml.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    stability = sub.add_parser("stability")
    stability.add_argument("--repeat", type=int, default=5)
    stability.add_argument("--profile", action="append", default=[])
    stability.add_argument(
        "--energy-atol", type=float, default=None,
        help="Override all profile-specific energy tolerances (Hartree).",
    )
    stability.add_argument(
        "--gradient-atol", type=float, default=None,
        help="Override all profile-specific gradient tolerances (Hartree/Bohr).",
    )
    all_parser = sub.add_parser("all")
    all_parser.add_argument("--repeat", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(default_config_path(args.config))
        payload: dict[str, object] = {"config": str(config.source_path)}
        passed = True
        if args.command in ("capabilities", "all"):
            capability = run_capability_probes(config, timeout=args.timeout)
            _print_capabilities(capability)
            payload["capabilities"] = [asdict(item) for item in capability]
            passed = passed and all(item.passed for item in capability)
        if args.command in ("stability", "all"):
            profiles = (
                tuple(args.profile)
                if args.command == "stability" and args.profile
                else DEFAULT_STABILITY_PROFILES
            )
            energy_atol = getattr(args, "energy_atol", None)
            gradient_atol = getattr(args, "gradient_atol", None)
            stability = run_stability_probes(
                config,
                profiles=profiles,
                repeats=args.repeat,
                timeout=args.timeout,
                energy_atol=energy_atol,
                gradient_atol=gradient_atol,
            )
            interleaving = run_interleave_probes(
                config,
                profiles=profiles,
                cycles=max(2, args.repeat),
                timeout=args.timeout,
                energy_atol=energy_atol,
                gradient_atol=gradient_atol,
            )
            _print_stability(stability)
            _print_interleaving(interleaving)
            payload["stability"] = [asdict(item) for item in stability]
            payload["interleaving"] = [asdict(item) for item in interleaving]
            passed = (
                passed
                and all(item.passed for item in stability)
                and all(item.passed for item in interleaving)
            )
        _write_json(args.json, payload)
        return 0 if passed else 2
    except Exception as exc:
        print(f"gau-maple-validation failed: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

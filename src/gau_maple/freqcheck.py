"""Command-line Hessian and frequency validation against a running server."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .client import evaluate_via_server
from .config import default_config_path, load_config
from .frequency import (
    atomic_masses_amu,
    compare_hessians,
    finite_difference_hessian,
    harmonic_frequency_analysis,
)
from .gaussian_io import parse_external_input, write_external_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gau-maple-freqcheck", allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="profiles.toml path")
    parser.add_argument("--profile", required=True, help="configured model profile")
    parser.add_argument("--input", type=Path, required=True, help="Gaussian derivative-order-2 .EIn")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--compare-fd", action="store_true", help="finite-difference server gradients")
    parser.add_argument("--step-bohr", type=float, default=1.0e-3)
    parser.add_argument("--max-rms-error", type=float, default=None)
    parser.add_argument("--max-abs-error", type=float, default=None)
    parser.add_argument("--write-eout", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _payload(
    *,
    profile: str,
    server: str,
    result: Any,
    analysis: Any,
    comparison: Any | None,
    fd: Any | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile": profile,
        "server": server,
        "energy_hartree": float(result.energy_hartree),
        "hessian_shape": list(result.hessian_hartree_per_bohr2.shape),
        "hessian_max_asymmetry": float(
            np.max(np.abs(result.hessian_hartree_per_bohr2 - result.hessian_hartree_per_bohr2.T))
        ),
        "external_mode_rank": int(analysis.external_mode_rank),
        "is_linear": bool(analysis.is_linear),
        "translation_residual": float(analysis.translation_residual),
        "imaginary_count": int(analysis.imaginary_count),
        "frequencies_cm1": [float(value) for value in analysis.frequencies_cm1],
        "intensity_note": (
            "IR/Raman intensities are unavailable because Gau_MAPLE currently emits "
            "zero dipole derivatives and polarizability placeholders."
        ),
    }
    if comparison is not None and fd is not None:
        payload["finite_difference"] = {
            "step_bohr": float(fd.step_bohr),
            "gradient_evaluations": int(fd.gradient_evaluations),
            "raw_max_asymmetry": float(fd.raw_max_asymmetry),
            "max_abs_error": float(comparison.max_abs_error),
            "rms_error": float(comparison.rms_error),
            "mean_abs_error": float(comparison.mean_abs_error),
            "reference_max_abs": float(comparison.reference_max_abs),
            "normalized_rms_error": float(comparison.normalized_rms_error),
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(default_config_path(args.config))
        profile = config.get_profile(args.profile)
        server = config.server_for_profile(args.profile)
        request = parse_external_input(args.input)
        if request.derivative_order != 2:
            parser.error("--input must be a derivative-order-2 Gaussian .EIn file")

        result, metadata = evaluate_via_server(
            request,
            server.socket_path,
            profile_name=profile.name,
            timeout=args.timeout,
            expect_server=server.name,
            expect_profile=profile.name,
        )
        assert result.hessian_hartree_per_bohr2 is not None

        masses = atomic_masses_amu(request.atomic_numbers)
        analysis = harmonic_frequency_analysis(
            result.hessian_hartree_per_bohr2,
            request.positions_bohr,
            masses,
        )

        fd = comparison = None
        if args.compare_fd:
            def evaluator(gradient_request):
                evaluated, _ = evaluate_via_server(
                    gradient_request,
                    server.socket_path,
                    profile_name=profile.name,
                    timeout=args.timeout,
                    expect_server=server.name,
                    expect_profile=profile.name,
                )
                return evaluated

            fd = finite_difference_hessian(request, evaluator, step_bohr=args.step_bohr)
            comparison = compare_hessians(
                result.hessian_hartree_per_bohr2,
                fd.hessian_hartree_per_bohr2,
            )

        if args.write_eout is not None:
            write_external_output(args.write_eout, request, result)

        payload = _payload(
            profile=profile.name,
            server=metadata.server_name,
            result=result,
            analysis=analysis,
            comparison=comparison,
            fd=fd,
        )
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("PASS Hessian request")
            print(f"profile: {profile.name}")
            print(f"server: {metadata.server_name}")
            print(f"energy_hartree: {result.energy_hartree:.12f}")
            print(f"hessian_shape: {result.hessian_hartree_per_bohr2.shape}")
            print(f"external_mode_rank: {analysis.external_mode_rank}")
            print(f"linear: {analysis.is_linear}")
            print(f"translation_residual: {analysis.translation_residual:.6e}")
            print(f"imaginary_count: {analysis.imaginary_count}")
            print("frequencies_cm-1:")
            for index, value in enumerate(analysis.frequencies_cm1, start=1):
                marker = "i" if value < 0.0 else ""
                print(f"  {index:3d}: {abs(value):12.4f}{marker}")
            print("NOTE: IR/Raman intensities are not provided by Gau_MAPLE.")
            if comparison is not None and fd is not None:
                print("finite_difference:")
                print(f"  step_bohr: {fd.step_bohr:.6e}")
                print(f"  gradient_evaluations: {fd.gradient_evaluations}")
                print(f"  raw_max_asymmetry: {fd.raw_max_asymmetry:.6e}")
                print(f"  max_abs_error: {comparison.max_abs_error:.6e}")
                print(f"  rms_error: {comparison.rms_error:.6e}")
                print(f"  normalized_rms_error: {comparison.normalized_rms_error:.6e}")

        failed = False
        if comparison is not None:
            if args.max_rms_error is not None and comparison.rms_error > args.max_rms_error:
                print(
                    f"FAIL: RMS Hessian error {comparison.rms_error:.6e} exceeds "
                    f"{args.max_rms_error:.6e}.",
                    file=sys.stderr,
                )
                failed = True
            if args.max_abs_error is not None and comparison.max_abs_error > args.max_abs_error:
                print(
                    f"FAIL: maximum Hessian error {comparison.max_abs_error:.6e} exceeds "
                    f"{args.max_abs_error:.6e}.",
                    file=sys.stderr,
                )
                failed = True
        return 2 if failed else 0
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"gau-maple-freqcheck failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

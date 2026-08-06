"""CLI health/status probe for a Gau_MAPLE multi-profile server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .client import ping_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gau-maple-ping", allow_abbrev=False)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--profile", help="optional profile to validate/select")
    parser.add_argument("--expect-server")
    parser.add_argument("--expect-profile")
    args = parser.parse_args(argv)
    profile = args.profile or args.expect_profile
    if args.profile and args.expect_profile and args.profile != args.expect_profile:
        print("gau-maple-ping failed: --profile and --expect-profile disagree", file=sys.stderr)
        return 2
    try:
        metadata = ping_server(
            args.socket,
            timeout=args.timeout,
            profile_name=profile,
            expect_server=args.expect_server,
            expect_profile=profile,
        )
    except Exception as exc:
        print(f"gau-maple-ping failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("PASS")
    print("server:", metadata.server_name)
    print("pid:", metadata.pid)
    print("request_count:", metadata.request_count)
    print("available_profiles:", ", ".join(metadata.available_profiles))
    if metadata.profile_name is not None:
        print("profile:", metadata.profile_name)
        print("model:", metadata.model)
        print("device:", metadata.device)
    print("profile_statuses:")
    for name in metadata.available_profiles:
        status = metadata.profile_statuses[name]
        line = (
            f"  {name}: state={status.get('preload_state')} "
            f"requests={status.get('request_count', 0)}"
        )
        if status.get("preload_error"):
            line += f" error={status['preload_error']}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

from gau_maple.freqcheck import build_parser


def test_freqcheck_parser() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config/profiles.toml",
            "--profile",
            "aimnet2",
            "--input",
            "water.EIn",
            "--compare-fd",
            "--step-bohr",
            "0.002",
        ]
    )
    assert args.config == Path("config/profiles.toml")
    assert args.profile == "aimnet2"
    assert args.input == Path("water.EIn")
    assert args.compare_fd
    assert args.step_bohr == 0.002

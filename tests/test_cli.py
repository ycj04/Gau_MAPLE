from __future__ import annotations

import pytest

from gau_maple.cli import build_parser, parse_key_value, parse_scalar, profile_from_namespace
from gau_maple.errors import ProfileError


def test_parse_scalar_conservative_types():
    assert parse_scalar("true") is True
    assert parse_scalar("false") is False
    assert parse_scalar("12") == 12
    assert parse_scalar("1.25e-3") == pytest.approx(1.25e-3)
    assert parse_scalar("uma-s-1p2") == "uma-s-1p2"
    assert parse_scalar("/absolute/model.pt") == "/absolute/model.pt"


def test_key_value_rejects_duplicates_and_missing_equals():
    with pytest.raises(ProfileError, match="KEY=VALUE"):
        parse_key_value(["bad"], label="--option")
    with pytest.raises(ProfileError, match="Duplicate"):
        parse_key_value(["x=1", "x=2"], label="--option")


def test_profile_from_cli_includes_plugin_and_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model",
            "esen_sm_conserving_all",
            "--module",
            "maple.function.calculator.esen_plugin",
            "--option",
            "model_path=/tmp/esen.pt",
            "--device",
            "cpu",
        ]
    )
    profile = profile_from_namespace(args)
    assert profile.model == "esen_sm_conserving_all"
    assert profile.model_options["module"] == "maple.function.calculator.esen_plugin"
    assert profile.model_options["model_path"] == "/tmp/esen.pt"


def test_socket_mode_does_not_require_model_and_rejects_direct_options():
    from gau_maple.cli import validate_external_mode

    parser = build_parser()
    args = parser.parse_args(["--socket", "/tmp/test.sock"])
    with pytest.raises(ProfileError, match="--profile"):
        validate_external_mode(args)

    args = parser.parse_args([
        "--socket", "/tmp/test.sock", "--profile", "aimnet2"
    ])
    assert validate_external_mode(args) == "socket"

    args = parser.parse_args(["--socket", "/tmp/test.sock", "--profile", "aimnet2", "--model", "aimnet2"])
    with pytest.raises(ProfileError, match="server"):
        validate_external_mode(args)


def test_direct_mode_still_requires_model():
    from gau_maple.cli import validate_external_mode

    parser = build_parser()
    args = parser.parse_args([])
    with pytest.raises(ProfileError, match="--model"):
        validate_external_mode(args)

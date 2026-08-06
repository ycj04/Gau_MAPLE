from __future__ import annotations

from pathlib import Path

import pytest

from gau_maple.errors import InvocationError
from gau_maple.invocation import parse_gaussian_invocation


def test_final_six_arguments_are_gaussian_owned(tmp_path):
    ein = tmp_path / "x.EIn"
    ein.write_text("dummy")
    inv = parse_gaussian_invocation(
        [
            "--model",
            "aimnet2",
            "--device",
            "cpu",
            "R",
            str(ein),
            str(tmp_path / "x.EOut"),
            str(tmp_path / "x.msg"),
            str(tmp_path / "x.fchk"),
            str(tmp_path / "x.mat"),
        ]
    )
    assert inv.option_argv == ("--model", "aimnet2", "--device", "cpu")
    assert inv.layer == "R"
    assert inv.input_path == ein


def test_too_few_gaussian_arguments_are_rejected():
    with pytest.raises(InvocationError, match="six final arguments"):
        parse_gaussian_invocation(["--model", "aimnet2", "R", "in", "out"])


def test_direct_mode_rejects_oniom_layer(tmp_path):
    ein = tmp_path / "x.EIn"
    ein.write_text("dummy")
    inv = parse_gaussian_invocation(
        ["M", str(ein), "out", "msg", "fchk", "mat"]
    )
    with pytest.raises(InvocationError, match="only.*layer 'R'"):
        inv.validate_direct_mode()


def test_direct_mode_rejects_missing_input(tmp_path):
    inv = parse_gaussian_invocation(
        ["R", str(tmp_path / "missing"), "out", "msg", "fchk", "mat"]
    )
    with pytest.raises(InvocationError, match="does not exist"):
        inv.validate_direct_mode()

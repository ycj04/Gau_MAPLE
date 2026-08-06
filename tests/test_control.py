from __future__ import annotations

from gau_maple.control import build_child_environment


def test_child_environment_clears_inherited_maple_plugin_variable():
    env = build_child_environment(
        {"MAPLE_CALCULATOR_PLUGINS": ""},
        base={"MAPLE_CALCULATOR_PLUGINS": "maple_mace_native", "PATH": "/bin"},
    )
    assert env["MAPLE_CALCULATOR_PLUGINS"] == ""
    assert env["PATH"] == "/bin"


def test_child_environment_applies_server_specific_values():
    env = build_child_environment(
        {"OMP_NUM_THREADS": "9", "EXTRA": "value"},
        base={"OMP_NUM_THREADS": "2"},
    )
    assert env["OMP_NUM_THREADS"] == "9"
    assert env["EXTRA"] == "value"

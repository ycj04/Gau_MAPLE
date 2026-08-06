from __future__ import annotations

import json
from pathlib import Path

from gau_maple.config import load_config
from gau_maple.doctor import (
    Diagnostic,
    _check_gaussian,
    _python_for_server,
    build_parser,
    main,
    run_diagnostics,
)


def write_config(tmp_path: Path, *, model_path: Path | None = None) -> Path:
    model_option = ""
    if model_path is not None:
        model_option = f'\n[profiles.alpha.model_options]\nmodel_path = "{model_path}"\n'
    path = tmp_path / "profiles.toml"
    path.write_text(
        f'''
[project]
runtime_dir = "{tmp_path}/runtime"

[profiles.alpha]
model = "aimnet2"
device = "cpu"
{model_option}
[servers.test_server]
executable = "/bin/true"
profiles = ["alpha"]
socket = "{{runtime_dir}}/test.sock"
pid_file = "{{runtime_dir}}/test.pid"
log = "{{runtime_dir}}/test.log"
stdout = "{{runtime_dir}}/test.stdout"
preload = false
startup_timeout = 2
shutdown_timeout = 2

[servers.test_server.environment]
MAPLE_CALCULATOR_PLUGINS = ""
''',
        encoding="utf-8",
    )
    return path


def test_diagnostic_flags():
    assert Diagnostic("x", "FAIL", "bad").failed is True
    assert Diagnostic("x", "WARN", "maybe").warned is True
    assert Diagnostic("x", "PASS", "ok").failed is False


def test_python_for_server_is_sibling_python(tmp_path):
    config = load_config(write_config(tmp_path))
    assert _python_for_server(config.servers["test_server"]) == Path("/bin/python")


def test_gaussian_check_pass_and_fail(tmp_path):
    good = tmp_path / "g16"
    good.write_text("#!/bin/sh\nexit 0\n")
    good.chmod(0o755)
    assert _check_gaussian(good)[0].status == "PASS"
    assert _check_gaussian(tmp_path / "missing")[0].status == "FAIL"


def test_run_diagnostics_without_server_or_import_probe(tmp_path, monkeypatch):
    config = load_config(write_config(tmp_path))
    monkeypatch.delenv("MAPLE_CALCULATOR_PLUGINS", raising=False)
    monkeypatch.setattr("gau_maple.doctor.importlib.util.find_spec", lambda name: type("Spec", (), {"origin": f"/{name}.py"})())
    result = run_diagnostics(
        config,
        probe_servers=False,
        probe_imports=False,
        gaussian=None,
    )
    names = {item.name for item in result}
    assert "config" in names
    assert "server:test_server:executable" in names
    assert "environment:MAPLE_CALCULATOR_PLUGINS" in names
    assert not any(item.failed for item in result)


def test_missing_model_path_is_failure(tmp_path):
    config = load_config(write_config(tmp_path, model_path=tmp_path / "missing.pt"))
    result = run_diagnostics(
        config,
        probe_servers=False,
        probe_imports=False,
    )
    item = next(x for x in result if x.name == "profile:alpha:model_path")
    assert item.status == "FAIL"


def test_parser_accepts_release_options():
    args = build_parser().parse_args(
        ["--skip-servers", "--skip-import-probes", "--strict", "--json", "report.json"]
    )
    assert args.skip_servers is True
    assert args.skip_import_probes is True
    assert args.strict is True
    assert args.json_path == Path("report.json")


def test_main_writes_json_report(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    monkeypatch.setattr("gau_maple.doctor.importlib.util.find_spec", lambda name: type("Spec", (), {"origin": f"/{name}.py"})())
    report = tmp_path / "doctor.json"
    monkeypatch.delenv("MAPLE_CALCULATOR_PLUGINS", raising=False)
    rc = main(
        [
            "--config",
            str(config_path),
            "--skip-servers",
            "--skip-import-probes",
            "--json",
            str(report),
        ]
    )
    assert rc == 0
    payload = json.loads(report.read_text())
    assert payload["config"] == str(config_path.absolute())
    assert payload["gau_maple_version"] == "0.10.0"
    assert payload["diagnostics"]

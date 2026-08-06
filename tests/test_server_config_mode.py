from __future__ import annotations

from pathlib import Path

from gau_maple import server as server_module


def test_server_main_loads_profiles_from_toml(tmp_path, monkeypatch):
    config = tmp_path / "profiles.toml"
    config.write_text(
        f'''
[project]
runtime_dir = "{tmp_path}/run"

[profiles.alpha]
model = "aimnet2"

[servers.one]
executable = "/bin/true"
profiles = ["alpha"]
socket = "{{runtime_dir}}/one.sock"
pid_file = "{{runtime_dir}}/one.pid"
log = "{{runtime_dir}}/one.log"
stdout = "{{runtime_dir}}/one.stdout"
preload = true
startup_timeout = 5
shutdown_timeout = 2

[servers.one.environment]
MAPLE_CALCULATOR_PLUGINS = ""
''',
        encoding="utf-8",
    )
    captured = {}

    def fake_run_server(profiles, socket_path, **kwargs):
        captured["profiles"] = profiles
        captured["socket"] = Path(socket_path)
        captured.update(kwargs)

    monkeypatch.setattr(server_module, "run_server", fake_run_server)
    monkeypatch.setenv("MAPLE_CALCULATOR_PLUGINS", "maple_mace_native")
    code = server_module.main(["--config", str(config), "--server", "one"])
    assert code == 0
    assert list(captured["profiles"]) == ["alpha"]
    assert captured["server_name"] == "one"
    assert captured["preload"] is True
    assert captured["socket"] == tmp_path / "run" / "one.sock"
    assert __import__("os").environ["MAPLE_CALCULATOR_PLUGINS"] == ""

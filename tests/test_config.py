from __future__ import annotations

from pathlib import Path

import pytest

from gau_maple.cli import build_parser, resolve_config_client, validate_external_mode
from gau_maple.config import default_config_path, load_config
from gau_maple.errors import ConfigError


def write_config(tmp_path: Path, *, extra: str = "") -> Path:
    path = tmp_path / "profiles.toml"
    path.write_text(
        f'''
[project]
runtime_dir = "{tmp_path}/run-{{user}}"

[profiles.alpha]
model = "aimnet2"
device = "cpu"

[profiles.alpha.model_options]
hessian = "analytic"

[profiles.beta]
model = "uma"
device = "cpu"

[profiles.beta.model_options]
size = "uma-s-1p2"
task = "omol"

[servers.maple_server]
executable = "/bin/true"
profiles = ["alpha"]
socket = "{{runtime_dir}}/maple.sock"
pid_file = "{{runtime_dir}}/maple.pid"
log = "{{runtime_dir}}/maple.log"
stdout = "{{runtime_dir}}/maple.stdout"
preload = true
startup_timeout = 5
shutdown_timeout = 2

[servers.maple_server.environment]
MAPLE_CALCULATOR_PLUGINS = ""

[servers.meta_server]
executable = "/bin/true"
profiles = ["beta"]
socket = "{{runtime_dir}}/meta.sock"
pid_file = "{{runtime_dir}}/meta.pid"
log = "{{runtime_dir}}/meta.log"
stdout = "{{runtime_dir}}/meta.stdout"
preload = false
startup_timeout = 5
shutdown_timeout = 2

[servers.meta_server.environment]
MAPLE_CALCULATOR_PLUGINS = ""
{extra}
''',
        encoding="utf-8",
    )
    return path


def test_load_config_routes_profiles_and_preserves_empty_environment(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.server_for_profile("alpha").name == "maple_server"
    assert config.server_for_profile("beta").name == "meta_server"
    assert config.servers["meta_server"].environment["MAPLE_CALCULATOR_PLUGINS"] == ""
    assert config.servers["maple_server"].preload is True
    assert config.servers["meta_server"].preload is False


def test_config_client_resolves_socket_and_server(tmp_path):
    path = write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["--config", str(path), "--profile", "beta"])
    resolve_config_client(args)
    assert args.socket == load_config(path).servers["meta_server"].socket_path
    assert args.expect_server == "meta_server"
    assert validate_external_mode(args) == "socket"


def test_config_client_rejects_wrong_server(tmp_path):
    path = write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        ["--config", str(path), "--profile", "beta", "--server", "maple_server"]
    )
    with pytest.raises(ConfigError, match="belongs to server"):
        resolve_config_client(args)


def test_duplicate_profile_assignment_is_rejected(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text()
    text = text.replace('profiles = ["beta"]', 'profiles = ["alpha", "beta"]')
    path.write_text(text)
    with pytest.raises(ConfigError, match="assigned to both"):
        load_config(path)


def test_unknown_profile_key_is_rejected(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text().replace('device = "cpu"\n\n[profiles.alpha.model_options]', 'device = "cpu"\nspeling = 1\n\n[profiles.alpha.model_options]', 1)
    path.write_text(text)
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(path)


def test_relative_runtime_path_is_rejected(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text().replace(f'runtime_dir = "{tmp_path}/run-{{user}}"', 'runtime_dir = "relative/run"')
    path.write_text(text)
    with pytest.raises(ConfigError, match="absolute path"):
        load_config(path)


def test_default_config_path_uses_environment(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.setenv("GAU_MAPLE_CONFIG", str(path))
    monkeypatch.chdir(tmp_path)
    assert default_config_path() == path.absolute()


def test_bare_profile_uses_gau_maple_config_environment(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    monkeypatch.setenv("GAU_MAPLE_CONFIG", str(path))
    parser = build_parser()
    args = parser.parse_args(["--profile", "alpha"])
    resolve_config_client(args)
    assert args.socket == load_config(path).servers["maple_server"].socket_path
    assert args.expect_server == "maple_server"
    assert validate_external_mode(args) == "socket"



def test_independent_electronic_state_policies_are_loaded(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text().replace(
        '[profiles.alpha]\nmodel = "aimnet2"\ndevice = "cpu"',
        '[profiles.alpha]\nmodel = "aimnet2"\ndevice = "cpu"\n'
        'charge_policy = "supported"\n'
        'multiplicity_policy = "singlet_only"',
    )
    path.write_text(text)
    config = load_config(path)
    alpha = config.profiles["alpha"]
    beta = config.profiles["beta"]
    assert alpha.charge_policy == "supported"
    assert alpha.multiplicity_policy == "singlet_only"
    assert beta.charge_policy == "calculator"
    assert beta.multiplicity_policy == "calculator"


def test_invalid_electronic_state_policy_is_rejected(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text().replace(
        '[profiles.alpha]\nmodel = "aimnet2"\ndevice = "cpu"',
        '[profiles.alpha]\nmodel = "aimnet2"\ndevice = "cpu"\n'
        'charge_policy = "sometimes"',
    )
    path.write_text(text)
    with pytest.raises(Exception, match="charge_policy"):
        load_config(path)

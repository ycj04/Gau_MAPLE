from __future__ import annotations

import os
from pathlib import Path

import pytest

from gau_maple.errors import ConfigError
from gau_maple.workflow import (
    extract_frequencies,
    job_run_succeeded,
    load_workflow,
    preflight_workflow,
    report_workflow,
    run_workflow,
    selected_job_names,
    summarize_gaussian_log,
)


def write_profiles(tmp_path: Path) -> Path:
    path = tmp_path / "profiles.toml"
    path.write_text(
        f'''
[project]
runtime_dir = "{tmp_path}/runtime"

[profiles.aimnet2]
model = "aimnet2"
device = "cpu"

[servers.maple_server]
executable = "/bin/true"
profiles = ["aimnet2"]
socket = "{{runtime_dir}}/maple.sock"
pid_file = "{{runtime_dir}}/maple.pid"
log = "{{runtime_dir}}/maple.log"
stdout = "{{runtime_dir}}/maple.stdout"
preload = false
startup_timeout = 5
shutdown_timeout = 2

[servers.maple_server.environment]
MAPLE_CALCULATOR_PLUGINS = ""
''',
        encoding="utf-8",
    )
    return path


def write_input(path: Path, profile: str = "aimnet2") -> None:
    path.write_text(
        "#p external='/bin/gau-maple --config /tmp/profiles.toml "
        f"--profile {profile}'\n\nTitle\n\n0 1\nH 0 0 0\n\n",
        encoding="utf-8",
    )


def write_workflow(tmp_path: Path, *, gaussian: str = "/bin/true") -> Path:
    profiles = write_profiles(tmp_path)
    input_a = tmp_path / "a.gjf"
    input_b = tmp_path / "b.gjf"
    write_input(input_a)
    write_input(input_b)
    path = tmp_path / "workflow.toml"
    path.write_text(
        f'''
[workflow]
working_dir = "{tmp_path}/runs"
gaussian_executable = "{gaussian}"
gau_maple_config = "{profiles}"
stop_on_failure = false
default_timeout = 10

[jobs.a]
kind = "sp"
input = "{input_a}"
output = "{tmp_path}/runs/a.log"
profile = "aimnet2"

[jobs.b]
kind = "freq"
input = "{input_b}"
output = "{tmp_path}/runs/b.log"
profile = "aimnet2"
depends_on = ["a"]
required_markers = ["Frequencies --", "Normal termination of Gaussian"]
expected_imaginary = 1
''',
        encoding="utf-8",
    )
    return path


def test_load_workflow_and_dependency_order(tmp_path):
    workflow = load_workflow(write_workflow(tmp_path))
    assert selected_job_names(workflow, ["b"]) == ("a", "b")
    assert workflow.jobs["b"].expected_imaginary == 1
    assert workflow.jobs["a"].output_path == tmp_path / "runs" / "a.log"


def test_workflow_cycle_is_rejected(tmp_path):
    path = write_workflow(tmp_path)
    text = path.read_text().replace('[jobs.a]\nkind = "sp"', '[jobs.a]\nkind = "sp"\ndepends_on = ["b"]')
    path.write_text(text)
    with pytest.raises(ConfigError, match="cycle"):
        load_workflow(path)


def test_preflight_checks_route_profile_without_gaussian_or_server(tmp_path):
    workflow = load_workflow(write_workflow(tmp_path))
    messages = preflight_workflow(
        workflow,
        require_gaussian=False,
        require_servers=False,
    )
    assert any("job=a" in line for line in messages)
    assert any("job=b" in line for line in messages)


def test_preflight_rejects_route_profile_mismatch(tmp_path):
    path = write_workflow(tmp_path)
    workflow = load_workflow(path)
    write_input(workflow.jobs["a"].input_path, profile="wrong")
    with pytest.raises(ConfigError, match="contains 'wrong'"):
        preflight_workflow(
            workflow,
            require_gaussian=False,
            require_servers=False,
        )


def test_frequency_extraction_and_imaginary_count(tmp_path):
    workflow = load_workflow(write_workflow(tmp_path))
    log = workflow.jobs["b"].output_path
    log.parent.mkdir(parents=True)
    log.write_text(
        " Frequencies --   -812.34  100.0  200.0\n"
        " Normal termination of Gaussian 16\n",
        encoding="utf-8",
    )
    assert extract_frequencies(log.read_text()) == (-812.34, 100.0, 200.0)
    summary = summarize_gaussian_log(workflow.jobs["b"])
    assert summary.passed
    assert summary.imaginary_count == 1


def test_missing_log_fails_report(tmp_path):
    workflow = load_workflow(write_workflow(tmp_path))
    summaries = report_workflow(workflow, selected=["a"])
    assert len(summaries) == 1
    assert not summaries[0].exists
    assert not summaries[0].passed


def test_run_workflow_with_fake_gaussian(tmp_path):
    fake = tmp_path / "fake-g16"
    fake.write_text(
        "#!/bin/sh\n"
        "cat >/dev/null\n"
        "echo ' Frequencies --   -500.0  100.0  200.0'\n"
        "echo ' Optimization completed.'\n"
        "echo ' Normal termination of Gaussian 16'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    workflow = load_workflow(write_workflow(tmp_path, gaussian=str(fake)))
    results = run_workflow(workflow)
    assert [result.name for result in results] == ["a", "b"]
    assert all(result.returncode == 0 for result in results)
    assert all(result.summary.passed for result in results)


def test_failed_dependency_skips_child_when_stop_on_failure_false(tmp_path):
    fake = tmp_path / "fake-g16"
    fake.write_text(
        "#!/bin/sh\ncat >/dev/null\necho ' Error termination request processed by link 9999.'\nexit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    workflow = load_workflow(write_workflow(tmp_path, gaussian=str(fake)))
    results = run_workflow(workflow)
    assert results[0].returncode == 1
    assert results[1].skipped
    assert "failed dependencies" in (results[1].skip_reason or "")


def test_unknown_job_selection_is_rejected(tmp_path):
    workflow = load_workflow(write_workflow(tmp_path))
    with pytest.raises(ConfigError, match="Unknown workflow job"):
        selected_job_names(workflow, ["missing"])


def test_nonzero_launcher_status_is_accepted_when_log_is_fully_valid(tmp_path):
    fake = tmp_path / "fake-g16-warning"
    fake.write_text(
        "#!/bin/sh\n"
        "cat >/dev/null\n"
        "echo ' Frequencies --   -500.0  100.0  200.0'\n"
        "echo ' Optimization completed.'\n"
        "echo ' Normal termination of Gaussian 16'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    workflow = load_workflow(write_workflow(tmp_path, gaussian=str(fake)))
    results = run_workflow(workflow)
    assert [result.returncode for result in results] == [1, 1]
    assert all(job_run_succeeded(result) for result in results)
    assert not results[1].skipped


def test_ts_workflow_splits_optimization_frequency_and_irc_dependencies(tmp_path):
    profiles = write_profiles(tmp_path)
    for name in ("ts", "freq", "forward", "reverse"):
        write_input(tmp_path / f"{name}.gjf")
    path = tmp_path / "ts_workflow.toml"
    path.write_text(
        f'''
[workflow]
working_dir = "{tmp_path}/runs"
gaussian_executable = "/bin/true"
gau_maple_config = "{profiles}"

[jobs.aimnet2_ts]
kind = "ts-opt"
input = "{tmp_path}/ts.gjf"
output = "{tmp_path}/runs/ts.log"
profile = "aimnet2"

[jobs.aimnet2_ts_freq]
kind = "ts-freq"
input = "{tmp_path}/freq.gjf"
output = "{tmp_path}/runs/freq.log"
profile = "aimnet2"
depends_on = ["aimnet2_ts"]

[jobs.aimnet2_irc_forward]
kind = "irc-forward"
input = "{tmp_path}/forward.gjf"
output = "{tmp_path}/runs/forward.log"
profile = "aimnet2"
depends_on = ["aimnet2_ts_freq"]

[jobs.aimnet2_irc_reverse]
kind = "irc-reverse"
input = "{tmp_path}/reverse.gjf"
output = "{tmp_path}/runs/reverse.log"
profile = "aimnet2"
depends_on = ["aimnet2_ts_freq"]
''',
        encoding="utf-8",
    )
    workflow = load_workflow(path)
    assert workflow.jobs["aimnet2_ts"].kind == "ts-opt"
    assert workflow.jobs["aimnet2_ts_freq"].depends_on == ("aimnet2_ts",)
    assert workflow.jobs["aimnet2_irc_forward"].depends_on == ("aimnet2_ts_freq",)
    assert workflow.jobs["aimnet2_irc_reverse"].depends_on == ("aimnet2_ts_freq",)

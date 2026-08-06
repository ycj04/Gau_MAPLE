"""Gaussian workflow runner and log validator.

This module deliberately keeps Gaussian as the workflow owner.  Gau_MAPLE only
starts/checks its persistent calculator servers, launches Gaussian, and audits
the resulting log files for the termination and chemistry-specific markers
requested in a strict TOML workflow definition.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 support
    import tomli as tomllib  # type: ignore[no-redef]

from .client import ping_server
from .config import GauMapleConfig, default_config_path, load_config
from .errors import ConfigError


_WORKFLOW_KEYS = {
    "working_dir",
    "gaussian_executable",
    "gau_maple_config",
    "stop_on_failure",
    "default_timeout",
}
_JOB_KEYS = {
    "kind",
    "input",
    "output",
    "profile",
    "enabled",
    "depends_on",
    "required_markers",
    "forbidden_markers",
    "expected_imaginary",
    "timeout",
    "environment",
}

_NORMAL_TERMINATION = "Normal termination of Gaussian"
_ERROR_TERMINATION = "Error termination"
_FREQUENCY_RE = re.compile(r"Frequencies\s+--\s+(.+)$")
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[DEde][-+]?\d+)?")


@dataclass(frozen=True, slots=True)
class GaussianJob:
    name: str
    kind: str
    input_path: Path
    output_path: Path
    profile: str
    enabled: bool = True
    depends_on: tuple[str, ...] = ()
    required_markers: tuple[str, ...] = (_NORMAL_TERMINATION,)
    forbidden_markers: tuple[str, ...] = (_ERROR_TERMINATION,)
    expected_imaginary: int | None = None
    timeout: float = 3600.0
    environment: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    source_path: Path
    working_dir: Path
    gaussian_executable: str
    gau_maple_config_path: Path
    stop_on_failure: bool
    default_timeout: float
    jobs: Mapping[str, GaussianJob]


@dataclass(frozen=True, slots=True)
class GaussianLogSummary:
    path: Path
    exists: bool
    normal_termination: bool
    error_termination: bool
    optimization_completed: bool
    stationary_point_found: bool
    frequencies_cm1: tuple[float, ...]
    imaginary_count: int
    elapsed_seconds: float | None
    required_markers_missing: tuple[str, ...]
    forbidden_markers_found: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class JobRunResult:
    name: str
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float
    skipped: bool
    skip_reason: str | None
    summary: GaussianLogSummary


def _unknown_keys(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {section}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}."
        )


def _nonempty(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{field_name} must not be empty.")
    return text


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be true or false, got {value!r}.")
    return value


def _positive_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a positive number.") from exc
    if number <= 0:
        raise ConfigError(f"{field_name} must be positive, got {number}.")
    return number


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_nonempty(value, field_name),)
    if not isinstance(value, list):
        raise ConfigError(f"{field_name} must be a string or TOML array of strings.")
    return tuple(_nonempty(item, field_name) for item in value)


def _string_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be a TOML table.")
    result: dict[str, str] = {}
    for key, item in value.items():
        name = _nonempty(key, f"{field_name} key")
        if not isinstance(item, (str, int, float, bool)):
            raise ConfigError(f"{field_name}.{name} must be a scalar.")
        if isinstance(item, bool):
            result[name] = "1" if item else "0"
        else:
            result[name] = str(item)
    return MappingProxyType(result)


def _expand_path(
    value: Any,
    *,
    field_name: str,
    variables: Mapping[str, str],
    base_dir: Path,
) -> Path:
    text = _nonempty(value, field_name)
    try:
        expanded = text.format_map(variables)
    except KeyError as exc:
        raise ConfigError(
            f"Unknown placeholder {{{exc.args[0]}}} in {field_name}."
        ) from exc
    path = Path(os.path.expandvars(os.path.expanduser(expanded)))
    if not path.is_absolute():
        path = base_dir / path
    return path.absolute()


def load_workflow(path: str | Path) -> WorkflowDefinition:
    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise ConfigError(f"Workflow file does not exist: {source}.")
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid workflow TOML in {source}: {exc}.") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Workflow TOML must contain top-level tables.")
    top_unknown = sorted(set(raw) - {"workflow", "jobs"})
    if top_unknown:
        raise ConfigError(
            f"Unknown workflow top-level section(s): {', '.join(top_unknown)}."
        )

    workflow_raw = raw.get("workflow", {})
    if not isinstance(workflow_raw, Mapping):
        raise ConfigError("[workflow] must be a TOML table.")
    _unknown_keys(workflow_raw, _WORKFLOW_KEYS, "[workflow]")

    project_dir = source.parent.parent
    variables = {
        "project_dir": str(project_dir),
        "workflow_dir": str(source.parent),
        "home": str(Path.home()),
        "user": os.environ.get("USER", "user"),
    }
    working_dir = _expand_path(
        workflow_raw.get("working_dir", "{project_dir}/workflow_runs"),
        field_name="workflow.working_dir",
        variables=variables,
        base_dir=source.parent,
    )
    variables = dict(variables)
    variables["working_dir"] = str(working_dir)

    gau_maple_config_path = _expand_path(
        workflow_raw.get(
            "gau_maple_config", "{project_dir}/config/profiles.toml"
        ),
        field_name="workflow.gau_maple_config",
        variables=variables,
        base_dir=source.parent,
    )
    gaussian_executable = _nonempty(
        workflow_raw.get("gaussian_executable", "g16"),
        "workflow.gaussian_executable",
    )
    stop_on_failure = _bool(
        workflow_raw.get("stop_on_failure", True),
        "workflow.stop_on_failure",
    )
    default_timeout = _positive_float(
        workflow_raw.get("default_timeout", 7200),
        "workflow.default_timeout",
    )

    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, Mapping) or not jobs_raw:
        raise ConfigError("At least one [jobs.NAME] table is required.")
    jobs: dict[str, GaussianJob] = {}
    for raw_name, raw_job in jobs_raw.items():
        name = _nonempty(raw_name, "job name")
        if not isinstance(raw_job, Mapping):
            raise ConfigError(f"[jobs.{name}] must be a TOML table.")
        _unknown_keys(raw_job, _JOB_KEYS, f"[jobs.{name}]")
        for required in ("kind", "input", "output", "profile"):
            if required not in raw_job:
                raise ConfigError(f"[jobs.{name}] is missing required key {required!r}.")
        expected_imaginary = raw_job.get("expected_imaginary")
        if expected_imaginary is not None:
            if isinstance(expected_imaginary, bool):
                raise ConfigError(
                    f"jobs.{name}.expected_imaginary must be a non-negative integer."
                )
            try:
                expected_imaginary = int(expected_imaginary)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"jobs.{name}.expected_imaginary must be a non-negative integer."
                ) from exc
            if expected_imaginary < 0:
                raise ConfigError(
                    f"jobs.{name}.expected_imaginary must be non-negative."
                )
        required_markers = _string_tuple(
            raw_job.get("required_markers", [_NORMAL_TERMINATION]),
            f"jobs.{name}.required_markers",
        )
        forbidden_markers = _string_tuple(
            raw_job.get("forbidden_markers", [_ERROR_TERMINATION]),
            f"jobs.{name}.forbidden_markers",
        )
        jobs[name] = GaussianJob(
            name=name,
            kind=_nonempty(raw_job["kind"], f"jobs.{name}.kind").lower(),
            input_path=_expand_path(
                raw_job["input"],
                field_name=f"jobs.{name}.input",
                variables=variables,
                base_dir=source.parent,
            ),
            output_path=_expand_path(
                raw_job["output"],
                field_name=f"jobs.{name}.output",
                variables=variables,
                base_dir=source.parent,
            ),
            profile=_nonempty(raw_job["profile"], f"jobs.{name}.profile"),
            enabled=_bool(raw_job.get("enabled", True), f"jobs.{name}.enabled"),
            depends_on=_string_tuple(
                raw_job.get("depends_on"), f"jobs.{name}.depends_on"
            ),
            required_markers=required_markers,
            forbidden_markers=forbidden_markers,
            expected_imaginary=expected_imaginary,
            timeout=_positive_float(
                raw_job.get("timeout", default_timeout), f"jobs.{name}.timeout"
            ),
            environment=_string_mapping(
                raw_job.get("environment"), f"jobs.{name}.environment"
            ),
        )

    for name, job in jobs.items():
        missing = sorted(set(job.depends_on) - set(jobs))
        if missing:
            raise ConfigError(
                f"Job {name!r} depends on unknown job(s): {', '.join(missing)}."
            )
        if name in job.depends_on:
            raise ConfigError(f"Job {name!r} cannot depend on itself.")

    _topological_order(jobs)
    return WorkflowDefinition(
        source_path=source,
        working_dir=working_dir,
        gaussian_executable=gaussian_executable,
        gau_maple_config_path=gau_maple_config_path,
        stop_on_failure=stop_on_failure,
        default_timeout=default_timeout,
        jobs=MappingProxyType(jobs),
    )


def _topological_order(jobs: Mapping[str, GaussianJob]) -> tuple[str, ...]:
    temporary: set[str] = set()
    permanent: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ConfigError(f"Workflow dependency cycle detected at job {name!r}.")
        temporary.add(name)
        for dependency in jobs[name].depends_on:
            visit(dependency)
        temporary.remove(name)
        permanent.add(name)
        ordered.append(name)

    for name in jobs:
        visit(name)
    return tuple(ordered)


def selected_job_names(
    workflow: WorkflowDefinition,
    selected: Iterable[str] | None = None,
) -> tuple[str, ...]:
    requested = tuple(str(item).strip() for item in (selected or ()) if str(item).strip())
    if requested:
        unknown = sorted(set(requested) - set(workflow.jobs))
        if unknown:
            raise ConfigError(
                f"Unknown workflow job(s): {', '.join(unknown)}. "
                f"Available: {', '.join(workflow.jobs)}."
            )
        closure: set[str] = set()

        def add_with_dependencies(name: str) -> None:
            if name in closure:
                return
            for dependency in workflow.jobs[name].depends_on:
                add_with_dependencies(dependency)
            closure.add(name)

        for name in requested:
            add_with_dependencies(name)
    else:
        closure: set[str] = set()

        def add_enabled_with_dependencies(name: str) -> None:
            if name in closure:
                return
            for dependency in workflow.jobs[name].depends_on:
                add_enabled_with_dependencies(dependency)
            closure.add(name)

        for name, job in workflow.jobs.items():
            if job.enabled:
                add_enabled_with_dependencies(name)
    return tuple(name for name in _topological_order(workflow.jobs) if name in closure)


def resolve_gaussian_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.absolute()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ConfigError(
                f"Gaussian executable is missing or not executable: {resolved}."
            )
        return resolved
    located = shutil.which(value)
    if located is None:
        raise ConfigError(
            f"Gaussian executable {value!r} was not found in PATH. Source the "
            "Gaussian environment or set workflow.gaussian_executable to an absolute path."
        )
    return Path(located).absolute()


def _extract_external_profile(input_text: str) -> str | None:
    match = re.search(r"--profile\s+(?:['\"]([^'\"]+)['\"]|([^\s'\"]+))", input_text)
    if not match:
        return None
    return (match.group(1) or match.group(2)).strip()


def preflight_workflow(
    workflow: WorkflowDefinition,
    *,
    selected: Iterable[str] | None = None,
    require_gaussian: bool = True,
    require_servers: bool = True,
) -> list[str]:
    messages: list[str] = []
    names = selected_job_names(workflow, selected)
    if require_gaussian:
        gaussian = resolve_gaussian_executable(workflow.gaussian_executable)
        messages.append(f"PASS gaussian_executable={gaussian}")
    else:
        messages.append("SKIP Gaussian executable check")

    if not workflow.gau_maple_config_path.is_file():
        raise ConfigError(
            f"Gau_MAPLE config file does not exist: {workflow.gau_maple_config_path}."
        )
    config = load_config(workflow.gau_maple_config_path)
    messages.append(f"PASS gau_maple_config={config.source_path}")

    checked_profiles: set[tuple[str, str]] = set()
    for name in names:
        job = workflow.jobs[name]
        if not job.input_path.is_file():
            raise ConfigError(f"Job {name!r} input does not exist: {job.input_path}.")
        text = job.input_path.read_text(encoding="utf-8", errors="replace")
        route_profile = _extract_external_profile(text)
        if route_profile != job.profile:
            raise ConfigError(
                f"Job {name!r} declares profile {job.profile!r}, but its Gaussian "
                f"External route contains {route_profile!r}."
            )
        server = config.server_for_profile(job.profile)
        messages.append(
            f"PASS job={name} kind={job.kind} profile={job.profile} "
            f"server={server.name} input={job.input_path}"
        )
        identity = (server.name, job.profile)
        if require_servers and identity not in checked_profiles:
            metadata = ping_server(
                server.socket_path,
                timeout=5.0,
                profile_name=job.profile,
                expect_server=server.name,
                expect_profile=job.profile,
            )
            status = metadata.profile_statuses.get(job.profile, {})
            if status.get("preload_state") == "failed":
                raise ConfigError(
                    f"Server {server.name!r} reports failed profile {job.profile!r}: "
                    f"{status.get('preload_error')}"
                )
            checked_profiles.add(identity)
            messages.append(
                f"PASS server={server.name} pid={metadata.pid} socket={server.socket_path}"
            )
    return messages


def extract_frequencies(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for line in text.splitlines():
        match = _FREQUENCY_RE.search(line)
        if not match:
            continue
        for token in _FLOAT_RE.findall(match.group(1)):
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                continue
    return tuple(values)


def summarize_gaussian_log(job: GaussianJob) -> GaussianLogSummary:
    path = job.output_path
    if not path.is_file():
        return GaussianLogSummary(
            path=path,
            exists=False,
            normal_termination=False,
            error_termination=False,
            optimization_completed=False,
            stationary_point_found=False,
            frequencies_cm1=(),
            imaginary_count=0,
            elapsed_seconds=None,
            required_markers_missing=job.required_markers,
            forbidden_markers_found=(),
            passed=False,
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    frequencies = extract_frequencies(text)
    imaginary_count = sum(value < 0.0 for value in frequencies)
    missing = tuple(marker for marker in job.required_markers if marker not in text)
    forbidden = tuple(marker for marker in job.forbidden_markers if marker in text)
    expected_ok = (
        job.expected_imaginary is None
        or imaginary_count == job.expected_imaginary
    )
    passed = not missing and not forbidden and expected_ok
    return GaussianLogSummary(
        path=path,
        exists=True,
        normal_termination=_NORMAL_TERMINATION in text,
        error_termination=_ERROR_TERMINATION in text,
        optimization_completed="Optimization completed" in text,
        stationary_point_found="Stationary point found" in text,
        frequencies_cm1=frequencies,
        imaginary_count=imaginary_count,
        elapsed_seconds=None,
        required_markers_missing=missing,
        forbidden_markers_found=forbidden,
        passed=passed,
    )


def run_gaussian_job(
    workflow: WorkflowDefinition,
    job: GaussianJob,
    gaussian_executable: Path,
) -> JobRunResult:
    workflow.working_dir.mkdir(parents=True, exist_ok=True)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = (str(gaussian_executable),)
    environment = os.environ.copy()
    environment.update(job.environment)
    started = time.monotonic()
    returncode: int | None = None
    try:
        with job.input_path.open("rb") as stdin, job.output_path.open("wb") as stdout:
            completed = subprocess.run(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                cwd=workflow.working_dir,
                env=environment,
                timeout=job.timeout,
                check=False,
            )
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        with job.output_path.open("ab") as handle:
            handle.write(
                f"\n[Gau_MAPLE WORKFLOW ERROR] Gaussian timed out after {job.timeout:g} s.\n".encode()
            )
        returncode = 124
    elapsed = time.monotonic() - started
    summary = summarize_gaussian_log(job)
    return JobRunResult(
        name=job.name,
        command=command,
        returncode=returncode,
        elapsed_seconds=elapsed,
        skipped=False,
        skip_reason=None,
        summary=summary,
    )


def job_run_succeeded(result: JobRunResult) -> bool:
    """Return the effective job status using Gaussian's own log as authority.

    Some local Gaussian launcher scripts return a non-zero shell status even
    after writing a complete ``Normal termination`` record.  Such a result is
    accepted only when every configured log validation passes and no Gaussian
    error termination is present.  The raw return code remains in reports.
    """
    if result.skipped or not result.summary.passed:
        return False
    if result.returncode == 0:
        return True
    return (
        result.summary.normal_termination
        and not result.summary.error_termination
    )


def run_workflow(
    workflow: WorkflowDefinition,
    *,
    selected: Iterable[str] | None = None,
) -> tuple[JobRunResult, ...]:
    gaussian = resolve_gaussian_executable(workflow.gaussian_executable)
    names = selected_job_names(workflow, selected)
    results: list[JobRunResult] = []
    status: dict[str, bool] = {}
    for name in names:
        job = workflow.jobs[name]
        failed_dependencies = [dep for dep in job.depends_on if not status.get(dep, False)]
        if failed_dependencies:
            summary = summarize_gaussian_log(job)
            result = JobRunResult(
                name=name,
                command=(str(gaussian),),
                returncode=None,
                elapsed_seconds=0.0,
                skipped=True,
                skip_reason=(
                    "failed dependencies: " + ", ".join(failed_dependencies)
                ),
                summary=summary,
            )
        else:
            result = run_gaussian_job(workflow, job, gaussian)
        results.append(result)
        status[name] = job_run_succeeded(result)
        if workflow.stop_on_failure and not status[name]:
            break
    return tuple(results)


def report_workflow(
    workflow: WorkflowDefinition,
    *,
    selected: Iterable[str] | None = None,
) -> tuple[GaussianLogSummary, ...]:
    return tuple(
        summarize_gaussian_log(workflow.jobs[name])
        for name in selected_job_names(workflow, selected)
    )


def _summary_line(name: str, summary: GaussianLogSummary) -> str:
    status = "PASS" if summary.passed else "FAIL"
    detail = [
        status,
        f"job={name}",
        f"log={summary.path}",
        f"normal={summary.normal_termination}",
        f"imaginary={summary.imaginary_count}",
    ]
    if summary.required_markers_missing:
        detail.append("missing=" + repr(summary.required_markers_missing))
    if summary.forbidden_markers_found:
        detail.append("forbidden=" + repr(summary.forbidden_markers_found))
    return " ".join(detail)


def _write_json_report(
    path: Path,
    workflow: WorkflowDefinition,
    results: Sequence[JobRunResult] | None,
    summaries: Sequence[tuple[str, GaussianLogSummary]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "workflow": str(workflow.source_path),
        "working_dir": str(workflow.working_dir),
        "jobs": [],
    }
    result_map = {result.name: result for result in (results or ())}
    for name, summary in summaries:
        result = result_map.get(name)
        payload["jobs"].append(
            {
                "name": name,
                "kind": workflow.jobs[name].kind,
                "profile": workflow.jobs[name].profile,
                "returncode": None if result is None else result.returncode,
                "elapsed_seconds": None if result is None else result.elapsed_seconds,
                "skipped": False if result is None else result.skipped,
                "skip_reason": None if result is None else result.skip_reason,
                "effective_success": None if result is None else job_run_succeeded(result),
                "summary": {
                    **asdict(summary),
                    "path": str(summary.path),
                },
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gau-maple-workflow", allow_abbrev=False)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="select one job; repeatable; dependencies are included automatically",
    )
    parser.add_argument("--json-report", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--skip-gaussian", action="store_true")
    preflight.add_argument("--skip-servers", action="store_true")
    sub.add_parser("run")
    sub.add_parser("report")
    sub.add_parser("all")
    sub.add_parser("list")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workflow = load_workflow(args.workflow)
        selected = args.job or None
        names = selected_job_names(workflow, selected)

        if args.command == "list":
            for name in names:
                job = workflow.jobs[name]
                print(
                    f"{name}: kind={job.kind} profile={job.profile} "
                    f"depends_on={','.join(job.depends_on) or '-'} input={job.input_path}"
                )
            return 0

        if args.command == "preflight":
            for line in preflight_workflow(
                workflow,
                selected=selected,
                require_gaussian=not args.skip_gaussian,
                require_servers=not args.skip_servers,
            ):
                print(line)
            return 0

        results: tuple[JobRunResult, ...] | None = None
        if args.command in ("run", "all"):
            if args.command == "all":
                for line in preflight_workflow(workflow, selected=selected):
                    print(line)
            results = run_workflow(workflow, selected=selected)
            for result in results:
                if result.skipped:
                    prefix = "SKIP"
                elif job_run_succeeded(result):
                    prefix = "PASS" if result.returncode == 0 else "WARN"
                else:
                    prefix = "FAIL"
                print(
                    f"{prefix} job={result.name} returncode={result.returncode} "
                    f"elapsed_s={result.elapsed_seconds:.3f} "
                    f"reason={result.skip_reason or '-'}"
                )
                print(_summary_line(result.name, result.summary))

        summaries_with_names = tuple(
            (name, summarize_gaussian_log(workflow.jobs[name])) for name in names
        )
        if args.command == "report":
            for name, summary in summaries_with_names:
                print(_summary_line(name, summary))

        if args.json_report:
            _write_json_report(
                args.json_report,
                workflow,
                results,
                summaries_with_names,
            )
            print(f"JSON report: {args.json_report}")

        if results is not None:
            return 0 if all(job_run_succeeded(result) for result in results) else 2
        return 0 if all(summary.passed for _, summary in summaries_with_names) else 2
    except Exception as exc:
        print(f"gau-maple-workflow failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Standard-library benchmark harness shared by every executable file."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_RESULT_PREFIX = "__FATQAT_BENCHMARK_RESULT__="
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = Path(tempfile.gettempdir()) / "fatqat-benchmark-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MATPLOTLIB_CACHE = _CACHE_ROOT / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """One configured operation and optional post-run diagnostics."""

    run: Callable[[], object]
    diagnostics: Callable[[object], Mapping[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Declarative benchmark case selected by one or more profiles."""

    name: str
    group: str
    profiles: frozenset[str]
    setup: Callable[[], PreparedCase]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    unstable: bool = False
    full_repeats: int | None = None

    @property
    def key(self) -> str:
        return f"{self.group}.{self.name}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone FatQat performance benchmarks."
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-micro", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--_worker-case", help=argparse.SUPPRESS)
    return parser


def _non_negative(value: int, label: str) -> int:
    if value < 0:
        raise SystemExit(f"{label} must be non-negative")
    return value


def _defaults(profile: str) -> tuple[int, int, float]:
    if profile == "quick":
        return 1, 5, 300.0
    return 2, 9, 1800.0


def _selected_cases(
    cases: Iterable[BenchmarkCase], profile: str, include_micro: bool
) -> tuple[BenchmarkCase, ...]:
    selected = []
    for case in cases:
        if profile not in case.profiles:
            continue
        if case.unstable and not include_micro:
            continue
        selected.append(case)
    keys = [case.key for case in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("benchmark case keys must be unique")
    return tuple(selected)


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata(module_path: Path) -> dict[str, Any]:
    for parent in (module_path.parent, *module_path.parents):
        if not (parent / ".git").exists():
            continue
        try:
            commit = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            dirty = subprocess.run(
                ["git", "-C", str(parent), "status", "--porcelain"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return {"root": str(parent), "commit": None, "dirty": None}
        return {
            "root": str(parent),
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    return {"root": None, "commit": None, "dirty": None}


def environment_metadata() -> dict[str, Any]:
    """Describe the interpreter and installed FatQat under measurement."""

    import fatqat

    module_path = Path(fatqat.__file__).resolve()
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "fatqat_version": _package_version("fatqat"),
        "fatqat_module": str(module_path),
        "fatqat_git": _git_metadata(module_path),
        "benchmark_git": _git_metadata(Path(__file__).resolve()),
        "numpy_version": _package_version("numpy"),
        "numba_version": _package_version("numba"),
    }


def _measure(call: Callable[[], object]) -> tuple[int, object]:
    start = time.perf_counter_ns()
    result = call()
    return time.perf_counter_ns() - start, result


def _run_worker(
    case: BenchmarkCase, *, profile: str, warmups: int, repeats: int
) -> dict[str, Any]:
    setup_start = time.perf_counter_ns()
    prepared = case.setup()
    setup_ns = time.perf_counter_ns() - setup_start

    cold_ns, last_result = _measure(prepared.run)
    for _ in range(warmups):
        last_result = prepared.run()

    samples = []
    for _ in range(repeats):
        sample, last_result = _measure(prepared.run)
        samples.append(sample)

    diagnostics = (
        dict(prepared.diagnostics(last_result))
        if prepared.diagnostics is not None
        else {}
    )
    summary = {
        "median_ns": int(statistics.median(samples)),
        "min_ns": min(samples),
        "max_ns": max(samples),
        "mean_ns": statistics.fmean(samples),
        "stdev_ns": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }
    return {
        "case": case.key,
        "group": case.group,
        "name": case.name,
        "profile": profile,
        "status": "ok",
        "unstable": case.unstable,
        "parameters": dict(case.parameters),
        "setup_ns": setup_ns,
        "cold_ns": cold_ns,
        "warmups": warmups,
        "repeats": repeats,
        "samples_ns": samples,
        "summary": summary,
        "diagnostics": diagnostics,
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }


def _worker_payload(
    cases: tuple[BenchmarkCase, ...], case_key: str, args: argparse.Namespace
) -> dict[str, Any]:
    case_by_key = {case.key: case for case in cases}
    if case_key not in case_by_key:
        return {
            "case": case_key,
            "status": "error",
            "error": {"type": "KeyError", "message": "unknown benchmark case"},
        }
    try:
        return _run_worker(
            case_by_key[case_key],
            profile=args.profile,
            warmups=args.warmups,
            repeats=args.repeats,
        )
    except Exception as exc:  # benchmark failures must be structured
        return {
            "case": case_key,
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "process_peak_rss_bytes": _peak_rss_bytes(),
        }


def _extract_worker_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX) :])
    raise ValueError("worker produced no structured result")


def _subprocess_output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _repeats_for_case(
    case: BenchmarkCase,
    *,
    profile: str,
    configured_repeats: int,
    repeats_were_supplied: bool,
) -> int:
    if repeats_were_supplied or profile != "full" or case.full_repeats is None:
        return configured_repeats
    return case.full_repeats


def _repeat_overrides(
    cases: Iterable[BenchmarkCase],
    *,
    profile: str,
    configured_repeats: int,
    repeats_were_supplied: bool,
) -> dict[str, int]:
    overrides = {}
    for case in cases:
        repeats = _repeats_for_case(
            case,
            profile=profile,
            configured_repeats=configured_repeats,
            repeats_were_supplied=repeats_were_supplied,
        )
        if repeats != configured_repeats:
            overrides[case.key] = repeats
    return overrides


def _run_subprocess(
    case: BenchmarkCase,
    *,
    entrypoint: Path,
    profile: str,
    warmups: int,
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(entrypoint),
        "--_worker-case",
        case.key,
        "--profile",
        profile,
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
    ]
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="fatqat-benchmark-numba-") as cache_dir:
        environment["NUMBA_CACHE_DIR"] = cache_dir
        try:
            completed = subprocess.run(
                command,
                cwd=_REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "case": case.key,
                "group": case.group,
                "name": case.name,
                "profile": profile,
                "status": "timeout",
                "timeout_seconds": timeout,
                "stdout": _subprocess_output_text(exc.stdout),
                "stderr": _subprocess_output_text(exc.stderr),
            }
    try:
        payload = _extract_worker_result(completed.stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "case": case.key,
            "group": case.group,
            "name": case.name,
            "profile": profile,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    if completed.stderr:
        payload["worker_stderr"] = completed.stderr
    payload["returncode"] = completed.returncode
    return payload


def _default_output(suite_name: str, profile: str) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return _REPOSITORY_ROOT / "results" / f"{timestamp}-{suite_name}-{profile}.json"


def _print_summary(results: list[dict[str, Any]]) -> None:
    print(f"{'case':58} {'status':9} {'cold ms':>12} {'median ms':>12}")
    print("-" * 96)
    for result in results:
        status = result["status"]
        cold = result.get("cold_ns")
        median = result.get("summary", {}).get("median_ns")
        cold_text = f"{cold / 1e6:.3f}" if isinstance(cold, (int, float)) else "-"
        median_text = f"{median / 1e6:.3f}" if isinstance(median, (int, float)) else "-"
        print(f"{result['case'][:58]:58} {status:9} {cold_text:>12} {median_text:>12}")


def main(cases: Iterable[BenchmarkCase], *, suite_name: str) -> int:
    """Run selected cases or execute one internal worker request."""

    all_cases = tuple(cases)
    args = _parser().parse_args()
    repeats_were_supplied = args.repeats is not None
    default_warmups, default_repeats, default_timeout = _defaults(args.profile)
    args.warmups = _non_negative(
        default_warmups if args.warmups is None else args.warmups, "warmups"
    )
    args.repeats = _non_negative(
        default_repeats if args.repeats is None else args.repeats, "repeats"
    )
    if args.repeats == 0:
        raise SystemExit("repeats must be positive")
    timeout = default_timeout if args.timeout is None else args.timeout
    if not math.isfinite(timeout) or timeout <= 0:
        raise SystemExit("timeout must be a positive finite number")

    if args._worker_case:
        payload = _worker_payload(all_cases, args._worker_case, args)
        print(_RESULT_PREFIX + json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] == "ok" else 1

    selected = _selected_cases(all_cases, args.profile, args.include_micro)
    if args.list:
        for case in selected:
            unstable = " [unstable]" if case.unstable else ""
            print(f"{case.key}{unstable}")
        return 0

    results = []
    entrypoint = Path(sys.argv[0]).resolve()
    for case in selected:
        repeats = _repeats_for_case(
            case,
            profile=args.profile,
            configured_repeats=args.repeats,
            repeats_were_supplied=repeats_were_supplied,
        )
        results.append(
            _run_subprocess(
                case,
                entrypoint=entrypoint,
                profile=args.profile,
                warmups=args.warmups,
                repeats=repeats,
                timeout=timeout,
            )
        )

    document = {
        "schema_version": 1,
        "suite": suite_name,
        "profile": args.profile,
        "environment": environment_metadata(),
        "configuration": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "case_repeat_overrides": _repeat_overrides(
                selected,
                profile=args.profile,
                configured_repeats=args.repeats,
                repeats_were_supplied=repeats_were_supplied,
            ),
            "timeout_seconds": timeout,
            "include_micro": args.include_micro,
            "numba_cache": "isolated_per_case",
        },
        "results": results,
    }
    output = (args.output or _default_output(suite_name, args.profile)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    _print_summary(results)
    print(f"\nresults: {output}")
    return 0 if all(result["status"] == "ok" for result in results) else 1

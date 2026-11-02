"""Compare selected warm workflows in two existing Python environments."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import statistics
import subprocess
import tempfile
from pathlib import Path

from benchmarks._harness import BenchmarkCase, _default_output, _run_subprocess

ROOT = Path(__file__).resolve().parent


def summarize(rounds, threshold=0.05, min_change_ms=0.0):
    """Bootstrap paired worker medians, not correlated calls within a worker."""
    if not rounds:
        return {"status": "error", "reason": "No paired rounds"}
    if any(
        pair[label].get("status") != "ok"
        for pair in rounds
        for label in ("base", "candidate")
    ):
        return {"status": "error", "reason": "A worker failed or timed out"}
    try:
        for pair in rounds:
            for label in ("base", "candidate"):
                samples = pair[label]["samples_ns"]
                if not samples or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in samples
                ):
                    return {"status": "error", "reason": "Invalid warm samples"}
        base = [statistics.median(pair["base"]["samples_ns"]) for pair in rounds]
        candidate = [
            statistics.median(pair["candidate"]["samples_ns"]) for pair in rounds
        ]
    except (KeyError, TypeError, statistics.StatisticsError):
        return {"status": "error", "reason": "Missing or invalid warm samples"}
    if any(not math.isfinite(v) or v <= 0 for v in base + candidate):
        return {"status": "error", "reason": "Non-positive or non-finite timing"}
    ratios = [c / b for b, c in zip(base, candidate)]
    changes = [c - b for b, c in zip(base, candidate)]
    result = {
        "base_ms": statistics.median(base) / 1e6,
        "candidate_ms": statistics.median(candidate) / 1e6,
        "ratio": statistics.median(ratios),
        "change_ms": statistics.median(changes) / 1e6,
    }
    if len(rounds) < 5:
        return {
            **result,
            "status": "inconclusive",
            "reason": "At least five paired rounds are needed",
        }
    rng = random.Random(17)
    ratio_samples, change_samples = [], []
    for _ in range(4000):
        indices = rng.choices(range(len(rounds)), k=len(rounds))
        ratio_samples.append(statistics.median(ratios[i] for i in indices))
        change_samples.append(statistics.median(changes[i] for i in indices) / 1e6)
    ratio_samples.sort()
    change_samples.sort()
    lower, upper = ratio_samples[100], ratio_samples[3899]
    abs_lower, abs_upper = change_samples[100], change_samples[3899]
    if lower > 1 + threshold and abs_lower > min_change_ms:
        status = "regression"
    elif upper <= 1 + threshold or abs_upper <= min_change_ms:
        status = "pass"
    else:
        status = "inconclusive"
    return {
        **result,
        "status": status,
        "ratio_interval_95": [lower, upper],
        "change_ms_interval_95": [abs_lower, abs_upper],
    }


def describe(python, entrypoint, args):
    command = [python, str(entrypoint), "--describe", "--profile", args.profile]
    for pattern in args.case:
        command += ["--case", pattern]
    for pattern in args.exclude:
        command += ["--exclude", pattern]
    if args.include_micro:
        command.append("--include-micro")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=args.timeout,
    )
    if completed.returncode:
        raise ValueError(
            f"Cannot enumerate {python}: {completed.stderr or completed.stdout}"
        )
    return json.loads(completed.stdout)


def check_manifests(base, candidate):
    if not base["cases"] or base["cases"] != candidate["cases"]:
        raise ValueError(
            "Base and candidate must select the same non-empty case definitions"
        )
    if len({case["case"] for case in base["cases"]}) != len(base["cases"]):
        raise ValueError("Duplicate case definitions")
    for key in (
        "python",
        "platform",
        "machine",
        "dependencies",
        "thread_environment",
        "benchmark_digest",
    ):
        if base["environment"][key] != candidate["environment"][key]:
            raise ValueError(
                f"Environment mismatch: {key}; use matching pins and controls"
            )


def exit_status(results):
    statuses = {r["summary"]["status"] for r in results}
    if "error" in statuses or not results:
        return 2
    if "regression" in statuses:
        return 1
    if "inconclusive" in statuses:
        return 3
    return 0


def save_report(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for result in document["results"]:
        summary = result["summary"]
        times = [
            f"{summary[k]:.3f}" if k in summary else "—"
            for k in ("base_ms", "candidate_ms")
        ]
        change = f"{(summary['ratio'] - 1) * 100:+.1f}%" if "ratio" in summary else "—"
        interval = summary.get("ratio_interval_95")
        bounds = (
            f"{(interval[0]-1)*100:+.1f}% to {(interval[1]-1)*100:+.1f}%"
            if interval
            else "—"
        )
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in [result["case"], summary["status"], *times, change, bounds]
            )
            + "</tr>"
        )
    error = html.escape(document.get("error", ""))
    settings = html.escape(json.dumps(document["configuration"], indent=2))
    environments = html.escape(json.dumps(document.get("environments", {}), indent=2))
    completion = (
        "Complete: all selected cases were attempted. See individual verdicts below."
        if document.get("complete", False)
        else "Incomplete: this report must not be treated as a passing comparison."
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FatQat local benchmark comparison</title>
<style>body{{font:16px/1.5 system-ui;margin:32px;color:#183243}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccd8dd}}th{{background:#edf4f6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.table{{overflow:auto}}</style>
<h1>FatQat local benchmark comparison</h1>
<p><strong>{completion}</strong></p>
<p>Warm timed operations; candidate / base. Each round uses fresh sequential workers in randomized order.
The interval is a paired bootstrap estimate, not a guarantee across machines or untested cases.
Confirm regressions with a longer rerun. Cold calls are recorded separately and are not gated.</p>
<p>{error}</p><div class="table"><table><thead><tr><th>Case</th><th>Verdict</th><th>Base ms</th><th>Candidate ms</th><th>Change</th><th>95% interval</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><a href="{html.escape(path.name, quote=True)}">Raw JSON, worker errors and timing samples</a></p>
<details><summary>Settings</summary><pre>{settings}</pre></details>
<details><summary>Environments and source fingerprints</summary><pre>{environments}</pre></details>
</html>"""
    path.with_suffix(".html").write_text(page, encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--base-python",
        required=True,
        help="Interpreter with baseline FatQat installed",
    )
    result.add_argument(
        "--candidate-python",
        required=True,
        help="Interpreter with candidate FatQat installed",
    )
    result.add_argument(
        "--runner", choices=("runner.py", "qasm_runner.py"), default="runner.py"
    )
    result.add_argument(
        "--profile", choices=("guard", "quick", "full"), default="guard"
    )
    result.add_argument("--case", action="append", default=[])
    result.add_argument("--exclude", action="append", default=[])
    result.add_argument("--include-micro", action="store_true")
    result.add_argument("--rounds", type=int, default=5)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--warmups", type=int, default=2)
    result.add_argument("--min-sample-time", type=float, default=0.05)
    result.add_argument("--timeout", type=float, default=300)
    result.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Fractional slowdown budget (default .05 = 5%%)",
    )
    result.add_argument(
        "--min-change-ms",
        type=float,
        default=0.0,
        help="Absolute slowdown budget per complete call",
    )
    result.add_argument("--seed", type=int, default=17)
    result.add_argument(
        "--output", type=Path, help="JSON output; adjacent HTML is also written"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    # Workers change directory to ROOT. Preserve paths relative to the caller,
    # without resolving venv executable symlinks to the system interpreter.
    for key in ("base_python", "candidate_python"):
        value = getattr(args, key)
        if "/" in value or "\\" in value:
            setattr(args, key, str(Path(value).expanduser().absolute()))
    if args.rounds < 1 or args.repeats < 1 or args.warmups < 0:
        raise SystemExit(
            "rounds/repeats must be positive; warmups must be non-negative"
        )
    for key in ("min_sample_time", "threshold", "min_change_ms", "timeout"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key == "timeout" and value == 0):
            raise SystemExit(f"Invalid {key}")
    output = (args.output or _default_output("comparison", args.profile)).resolve()
    if output.suffix != ".json":
        raise SystemExit("output must have a .json suffix")
    if output.exists() or output.with_suffix(".html").exists():
        raise SystemExit(
            "Output already exists; choose a new path to preserve prior evidence"
        )
    entrypoint = ROOT / args.runner
    document = {
        "schema_version": 1,
        "configuration": {**vars(args), "output": str(output)},
        "results": [],
        "complete": False,
    }
    try:
        manifests = {
            label: describe(python, entrypoint, args)
            for label, python in (
                ("base", args.base_python),
                ("candidate", args.candidate_python),
            )
        }
        document["environments"] = {k: v["environment"] for k, v in manifests.items()}
        check_manifests(manifests["base"], manifests["candidate"])
        document["selection"] = manifests["base"]["cases"]
        rng = random.Random(args.seed)
        # Reuse disk caches within each revision, never between revisions. Each
        # worker still has a fresh process and its own warm-up before sampling.
        with tempfile.TemporaryDirectory(prefix="fatqat-compare-") as temp:
            caches = {label: str(Path(temp) / label) for label in ("base", "candidate")}
            for cache in caches.values():
                Path(cache).mkdir()
            for spec in document["selection"]:
                group, name = spec["case"].rsplit(".", 1)
                case = BenchmarkCase(
                    name, group, frozenset({args.profile}), lambda: None
                )
                rounds = []
                result = {
                    **spec,
                    "rounds": rounds,
                    "summary": {"status": "inconclusive", "reason": "Not yet measured"},
                }
                document["results"].append(result)
                save_report(output, document)
                for index in range(args.rounds):
                    order = rng.sample(["base", "candidate"], 2)
                    pair = {"order": order}
                    for label in order:
                        print(
                            f"{case.key}: round {index+1}/{args.rounds}, {label}",
                            flush=True,
                        )
                        pair[label] = _run_subprocess(
                            case,
                            entrypoint=entrypoint,
                            profile=args.profile,
                            warmups=args.warmups,
                            repeats=args.repeats,
                            timeout=args.timeout,
                            python=getattr(args, label + "_python"),
                            min_sample_time=args.min_sample_time,
                            cache_dir=caches[label],
                        )
                    rounds.append(pair)
                    result["summary"] = summarize(
                        rounds, args.threshold, args.min_change_ms
                    )
                    save_report(output, document)
                    if any(pair[label]["status"] != "ok" for label in order):
                        break
                print(f"  {result['summary']['status']}", flush=True)
        document["complete"] = True
        code = exit_status(document["results"])
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        document["error"] = str(exc)
        code = 2
    document["exit_code"] = code
    save_report(output, document)
    print(f"Report: {output.with_suffix('.html')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

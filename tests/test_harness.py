from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks._harness import (
    BenchmarkCase,
    PreparedCase,
    _git_metadata,
    _repeat_overrides,
    _repeats_for_case,
    _run_subprocess,
    _selected_cases,
    _worker_payload,
    _run_worker,
)


def _prepared_noop() -> PreparedCase:
    return PreparedCase(lambda: None)


class HarnessTests(unittest.TestCase):
    def test_includes_are_or_and_exclusions_win(self):
        cases = tuple(
            BenchmarkCase(
                name=n, group="test", profiles=frozenset({"full"}), setup=_prepared_noop
            )
            for n in ("sweep1", "sweep8", "state")
        )
        self.assertEqual(
            _selected_cases(cases, "full", False, ["*sweep*", "*state"], ["*8"]),
            (cases[0], cases[2]),
        )
        self.assertEqual(_selected_cases(cases, "quick", False, ["*"]), ())
        self.assertEqual(_selected_cases(cases, "full", False, ["typo"]), ())

    def test_minimum_sample_time_records_calls_and_per_call_times(self):
        ticks = iter(range(0, 100_000_000, 1_000_000))
        case = BenchmarkCase("noop", "test", frozenset({"quick"}), _prepared_noop)
        with mock.patch(
            "benchmarks._harness.time.perf_counter_ns", side_effect=lambda: next(ticks)
        ):
            result = _run_worker(
                case, profile="quick", warmups=0, repeats=2, min_sample_time=0.003
            )
        self.assertEqual(result["sample_calls"], [3, 3])
        self.assertEqual(result["samples_ns"], [1_000_000, 1_000_000])

    def test_nonzero_worker_exit_cannot_report_success(self):
        case = BenchmarkCase("noop", "test", frozenset({"quick"}), _prepared_noop)
        completed = subprocess.CompletedProcess(
            [], 1, '__FATQAT_BENCHMARK_RESULT__={"status":"ok"}\n', ""
        )
        with mock.patch("benchmarks._harness.subprocess.run", return_value=completed):
            result = _run_subprocess(
                case,
                entrypoint=Path("runner.py"),
                profile="quick",
                warmups=0,
                repeats=1,
                timeout=1,
            )
        self.assertEqual(result["status"], "error")

    def test_profile_and_unstable_selection(self):
        stable = BenchmarkCase(
            name="stable",
            group="test",
            profiles=frozenset({"quick", "full"}),
            setup=_prepared_noop,
        )
        full = BenchmarkCase(
            name="full",
            group="test",
            profiles=frozenset({"full"}),
            setup=_prepared_noop,
        )
        unstable = BenchmarkCase(
            name="unstable",
            group="test",
            profiles=frozenset({"quick", "full"}),
            setup=_prepared_noop,
            unstable=True,
        )

        self.assertEqual(
            _selected_cases((stable, full, unstable), "quick", False), (stable,)
        )
        self.assertEqual(
            _selected_cases((stable, full, unstable), "quick", True),
            (stable, unstable),
        )

    def test_duplicate_case_keys_are_rejected(self):
        first = BenchmarkCase(
            name="same",
            group="test",
            profiles=frozenset({"quick"}),
            setup=_prepared_noop,
        )
        second = BenchmarkCase(
            name="same",
            group="test",
            profiles=frozenset({"quick"}),
            setup=_prepared_noop,
        )
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _selected_cases((first, second), "quick", False)

    def test_full_repeat_override_is_explicit(self):
        case = BenchmarkCase(
            name="compiler",
            group="test",
            profiles=frozenset({"full"}),
            setup=_prepared_noop,
            full_repeats=5,
        )
        self.assertEqual(
            _repeats_for_case(
                case,
                profile="full",
                configured_repeats=9,
                repeats_were_supplied=False,
            ),
            5,
        )
        self.assertEqual(
            _repeat_overrides(
                (case,),
                profile="full",
                configured_repeats=9,
                repeats_were_supplied=False,
            ),
            {"test.compiler": 5},
        )
        self.assertEqual(
            _repeat_overrides(
                (case,),
                profile="full",
                configured_repeats=7,
                repeats_were_supplied=True,
            ),
            {},
        )

    def test_timeout_bytes_are_json_serializable(self):
        case = BenchmarkCase(
            name="timeout",
            group="test",
            profiles=frozenset({"quick"}),
            setup=_prepared_noop,
        )
        timeout = subprocess.TimeoutExpired(
            cmd=["worker"],
            timeout=0.1,
            output=b"partial stdout \xff",
            stderr=b"partial stderr \xfe",
        )
        with mock.patch("benchmarks._harness.subprocess.run", side_effect=timeout):
            result = _run_subprocess(
                case,
                entrypoint=Path("runner.py"),
                profile="quick",
                warmups=0,
                repeats=1,
                timeout=0.1,
            )

        self.assertEqual(result["status"], "timeout")
        self.assertIsInstance(result["stdout"], str)
        self.assertIsInstance(result["stderr"], str)
        json.dumps(result)

    def test_missing_git_executable_degrades_to_null_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            module_path = root / "package" / "module.py"
            with mock.patch(
                "benchmarks._harness.subprocess.run", side_effect=FileNotFoundError
            ):
                metadata = _git_metadata(module_path)

        self.assertEqual(
            metadata,
            {"root": str(root), "commit": None, "dirty": None},
        )

    def test_worker_failure_is_structured(self):
        def fail():
            raise RuntimeError("intentional")

        case = BenchmarkCase(
            name="failure",
            group="test",
            profiles=frozenset({"quick"}),
            setup=lambda: PreparedCase(fail),
        )
        arguments = mock.Mock(profile="quick", warmups=0, repeats=1)
        payload = _worker_payload((case,), case.key, arguments)

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["type"], "RuntimeError")
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()

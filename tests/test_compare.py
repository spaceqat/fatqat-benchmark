"""Regression verdicts and the two-environment command contract."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compare


def worker(time_ns):
    return {"status": "ok", "samples_ns": [time_ns] * 5}


def rounds(ratios):
    return [
        {"base": worker(1_000_000), "candidate": worker(r * 1_000_000)} for r in ratios
    ]


def manifest():
    return {
        "cases": [{"case": "test.example", "parameters": {}, "unstable": False}],
        "environment": {
            "python": "3.13",
            "platform": "test",
            "machine": "test",
            "dependencies": {},
            "thread_environment": {},
            "benchmark_digest": "same",
        },
    }


class ComparisonTests(unittest.TestCase):
    def test_clear_change_noise_and_insufficient_evidence(self):
        self.assertEqual(compare.summarize(rounds([1] * 7))["status"], "pass")
        self.assertEqual(compare.summarize(rounds([1.2] * 7))["status"], "regression")
        self.assertEqual(
            compare.summarize(rounds([0.8, 1.4, 0.9, 1.2, 1.1]))["status"],
            "inconclusive",
        )
        self.assertEqual(compare.summarize(rounds([1.2] * 3))["status"], "inconclusive")
        self.assertEqual(
            compare.summarize(rounds([1.2] * 7), min_change_ms=0.3)["status"], "pass"
        )

    def test_failures_and_invalid_samples_cannot_pass(self):
        data = rounds([1] * 5)
        for failure in (
            {"status": "timeout"},
            {"status": "error"},
            worker(float("nan")),
            worker(0),
            {"status": "ok", "samples_ns": []},
            *(
                {"status": "ok", "samples_ns": [1_000_000] * 4 + [bad]}
                for bad in (float("nan"), float("inf"), 0, -1, "invalid", True)
            ),
        ):
            with self.subTest(failure=failure):
                data[0]["candidate"] = failure
                self.assertEqual(compare.summarize(data)["status"], "error")
        self.assertEqual(compare.summarize([])["status"], "error")

    def test_relative_interpreters_are_resolved_before_worker_directory_change(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.chdir(directory),
            mock.patch.object(compare, "describe", return_value=manifest()) as describe,
            mock.patch.object(
                compare, "_run_subprocess", return_value=worker(100)
            ) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            compare.main(
                [
                    "--base-python",
                    "base/python",
                    "--candidate-python",
                    "candidate/python",
                    "--rounds",
                    "1",
                    "--output",
                    "report.json",
                ]
            )
            expected = {
                str(Path(p).absolute()) for p in ("base/python", "candidate/python")
            }
            self.assertEqual(
                {call.args[0] for call in describe.call_args_list}, expected
            )
            self.assertEqual(
                {call.kwargs["python"] for call in run.call_args_list}, expected
            )

    def test_partial_html_is_marked_incomplete_even_with_passing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            document = {
                "configuration": {},
                "complete": False,
                "results": [{"case": "test.example", "summary": {"status": "pass"}}],
            }
            compare.save_report(output, document)
            self.assertIn("Incomplete:", output.with_suffix(".html").read_text())
            document["complete"] = True
            compare.save_report(output, document)
            self.assertNotIn("Incomplete:", output.with_suffix(".html").read_text())

    def test_environment_and_case_mismatch_fail_closed(self):
        base = manifest()
        compare.check_manifests(base, copy.deepcopy(base))
        for field in (
            "python",
            "dependencies",
            "benchmark_digest",
            "thread_environment",
        ):
            candidate = copy.deepcopy(base)
            candidate["environment"][field] = "different"
            with self.assertRaisesRegex(ValueError, field):
                compare.check_manifests(base, candidate)
        for cases in ([], [{"case": "missing"}], base["cases"] * 2):
            candidate = copy.deepcopy(base)
            candidate["cases"] = cases
            with self.assertRaises(ValueError):
                compare.check_manifests(base, candidate)

    def test_command_uses_both_interpreters_and_preserves_raw_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            calls = []

            def run(_case, **kwargs):
                calls.append(kwargs)
                return worker(
                    1_000_000 if kwargs["python"] == "base-python" else 1_200_000
                )

            with (
                mock.patch.object(compare, "describe", return_value=manifest()),
                mock.patch.object(compare, "_run_subprocess", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = compare.main(
                    [
                        "--base-python",
                        "base-python",
                        "--candidate-python",
                        "pr-python",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertEqual(len(calls), 10)
            self.assertEqual({c["python"] for c in calls}, {"base-python", "pr-python"})
            self.assertEqual(len({c["cache_dir"] for c in calls}), 2)
            document = json.loads(output.read_text())
            self.assertTrue(document["complete"])
            self.assertEqual(len(document["results"][0]["rounds"]), 5)
            self.assertIn("regression", output.with_suffix(".html").read_text())
            with self.assertRaisesRegex(SystemExit, "already exists"):
                compare.main(
                    [
                        "--base-python",
                        "base",
                        "--candidate-python",
                        "pr",
                        "--output",
                        str(output),
                    ]
                )

    def test_preflight_failure_still_writes_error_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            with (
                mock.patch.object(
                    compare, "describe", side_effect=ValueError("unavailable")
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = compare.main(
                    [
                        "--base-python",
                        "base",
                        "--candidate-python",
                        "pr",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertFalse(json.loads(output.read_text())["complete"])
            self.assertIn("unavailable", output.with_suffix(".html").read_text())

    def test_exit_precedence_and_html_escaping(self):
        self.assertEqual(compare.exit_status([]), 2)
        for expected, statuses in (
            (2, ["error", "regression"]),
            (1, ["pass", "regression"]),
            (3, ["pass", "inconclusive"]),
            (0, ["pass"]),
        ):
            self.assertEqual(
                compare.exit_status([{"summary": {"status": s}} for s in statuses]),
                expected,
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            compare.save_report(
                output, {"configuration": {}, "results": [], "error": "<script>"}
            )
            self.assertIn("&lt;script&gt;", output.with_suffix(".html").read_text())


if __name__ == "__main__":
    unittest.main()

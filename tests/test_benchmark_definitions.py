from __future__ import annotations

import hashlib
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from fatqat.qasm import from_qasm

import qasm_runner
import runner
from benchmarks import _qasm_corpus
from benchmarks._harness import _selected_cases
from benchmarks._qasm_corpus import load_qasm_fixture, validate_fixture_metadata
from benchmarks.bench_simulator_shots import CASES as SHOT_CASES


class BenchmarkDefinitionTests(unittest.TestCase):
    def test_expected_profile_sizes(self):
        self.assertEqual(len(_selected_cases(runner.CASES, "quick", False)), 10)
        self.assertEqual(len(_selected_cases(runner.CASES, "full", False)), 28)
        self.assertEqual(len(_selected_cases(qasm_runner.CASES, "quick", False)), 8)
        self.assertEqual(len(_selected_cases(qasm_runner.CASES, "full", False)), 16)

    def test_parallel_worker_metadata_matches_strategy(self):
        cases = {
            case.parameters["strategy"]: case
            for case in SHOT_CASES
            if case.group == "simulator.parallel"
        }
        self.assertEqual(cases["serial"].parameters["workers"], 1)
        self.assertEqual(cases["threads"].parameters["workers"], 2)
        self.assertEqual(cases["processes"].parameters["workers"], 2)

    def test_qasm_manifest_and_case_metadata_agree(self):
        manifest_path = (
            Path(_qasm_corpus.__file__).resolve().parents[1]
            / "corpora"
            / "qasm"
            / "manifest.toml"
        )
        document = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ids = {entry["id"] for entry in document["fixture"]}
        case_ids = {case.parameters["fixture_id"] for case in qasm_runner.CASES}
        self.assertEqual(manifest_ids, case_ids)

        for case in qasm_runner.CASES:
            fixture = load_qasm_fixture(case.parameters["fixture_id"])
            qubits = case.parameters.get("qubits", case.parameters.get("atoms"))
            validate_fixture_metadata(
                fixture,
                profiles=case.profiles,
                qubits=qubits,
                qasm_format=case.parameters["format"],
            )

    def test_qasm_pairs_import_with_equal_node_counts(self):
        pairs: dict[str, dict[str, int]] = {}
        for case in qasm_runner.CASES:
            fixture = load_qasm_fixture(case.parameters["fixture_id"])
            program = from_qasm(fixture.source)
            stem = fixture.fixture_id.rsplit("_qasm", 1)[0]
            pairs.setdefault(stem, {})[fixture.qasm_format] = len(program.dag().nodes)

        self.assertEqual(len(pairs), 8)
        for formats in pairs.values():
            self.assertEqual(set(formats), {"openqasm2", "openqasm3"})
            self.assertEqual(len(set(formats.values())), 1)

    def test_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus_root = Path(directory)
            source_path = corpus_root / "fixture.qasm"
            source_path.write_text("OPENQASM 2.0;\n", encoding="utf-8")
            manifest = self._write_manifest(
                corpus_root,
                path="fixture.qasm",
                digest="0" * 64,
            )
            with self._patched_corpus(corpus_root, manifest):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_qasm_fixture("fixture")

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_root = root / "corpus"
            corpus_root.mkdir()
            outside = root / "outside.qasm"
            outside.write_text("OPENQASM 2.0;\n", encoding="utf-8")
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            manifest = self._write_manifest(
                corpus_root,
                path="../outside.qasm",
                digest=digest,
            )
            with self._patched_corpus(corpus_root, manifest):
                with self.assertRaisesRegex(ValueError, "escapes the corpus root"):
                    load_qasm_fixture("fixture")

    @staticmethod
    def _write_manifest(corpus_root: Path, *, path: str, digest: str) -> Path:
        manifest = corpus_root / "manifest.toml"
        manifest.write_text(
            textwrap.dedent(f"""\
                schema_version = 1

                [[fixture]]
                id = "fixture"
                path = "{path}"
                format = "openqasm2"
                profiles = ["quick", "full"]
                qubits = 1
                tags = ["test"]
                sha256 = "{digest}"
                origin = "test"
                license = "Apache-2.0"
                generated_by = "test"
                exporter_fatqat_commit = "test"
                workload = "test"
                workload_parameters = {{ width = 1 }}
                """),
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _patched_corpus(corpus_root: Path, manifest: Path):
        return mock.patch.multiple(
            _qasm_corpus,
            _CORPUS_ROOT=corpus_root,
            _MANIFEST=manifest,
        )


if __name__ == "__main__":
    unittest.main()

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
from benchmarks.bench_simulator_state import _state_setup


class BenchmarkDefinitionTests(unittest.TestCase):
    def test_fusion_controls_are_explicit_and_other_cases_keep_default(self):
        for fusion in (None, False, True):
            with mock.patch(
                "benchmarks.bench_simulator_state.fq.simulator.Simulator"
            ) as simulator:
                prepared = _state_setup("density_matrix", "numba", 2, 1, fusion=fusion)
                prepared.run()
                config = simulator.return_value.run.call_args.kwargs[
                    "simulation_config"
                ]
                if fusion is None:
                    self.assertNotIn("fusion", config)
                else:
                    self.assertIs(config["fusion"], fusion)

    def test_expected_profile_sizes(self):
        self.assertEqual(len(_selected_cases(runner.CASES, "quick", False)), 45)
        self.assertEqual(len(_selected_cases(runner.CASES, "full", False)), 322)
        self.assertEqual(len(_selected_cases(runner.CASES, "guard", False)), 38)
        self.assertEqual(len(_selected_cases(qasm_runner.CASES, "quick", False)), 8)
        self.assertEqual(len(_selected_cases(qasm_runner.CASES, "full", False)), 16)

    def test_guard_reuses_numba_catalog_and_keeps_targeted_probes_in_full(self):
        guard = {
            case.key: case for case in _selected_cases(runner.CASES, "guard", False)
        }
        full = {case.key: case for case in _selected_cases(runner.CASES, "full", False)}
        for family, sizes in (
            ("ghz_chain", (5, 15, 24)),
            ("qft", (5, 15, 24)),
            ("qaoa_maxcut", (5, 15, 24)),
            ("efficient_su2_ring", (5, 15, 20)),
            ("vqe_brickwall", (5, 15, 20)),
            ("real_amplitudes", (5, 15, 20)),
        ):
            for size in sizes:
                key = f"simulator.catalog.{family}_numba_{size}q"
                self.assertIn(key, guard)
                self.assertIs(guard[key].setup, full[key].setup)
                self.assertNotIn(f"simulator.catalog.{family}_numpy_{size}q", guard)
        for name in (
            "sweep_8q_3l_5p_1rows",
            "sweep_13q_1l_5p_2049rows_auto4",
            "sweep_structure_8q_64rows",
            "sweep_8q_16l_128p_64rows",
            "sweep_amplitude_64rows_2shots",
            "sweep_amplitude_256rows_2shots",
            "sweep_amplitude_64rows_128shots_threads2",
            "local_ccx_14q_64g_threads8",
        ):
            key = f"simulator.guard.{name}"
            self.assertIn(key, full)
            self.assertNotIn(key, guard)

    def test_guard_covers_all_matrix_methods_and_sampling_workflows(self):
        guard = {
            case.key: case for case in _selected_cases(runner.CASES, "guard", False)
        }
        full = {case.key: case for case in _selected_cases(runner.CASES, "full", False)}
        for key in (
            "simulator.catalog.method_unitary_numba_8q",
            "simulator.catalog.small_superop_numba_4q",
            "simulator.sampling.fixed_state_statevector_numba_12q_65536s",
            "simulator.sampling.terminal_noise_density_matrix_numba_8q_32s",
            "simulator.sampling.dynamic_noise_density_matrix_numba_5q_32s",
            "simulator.state.dm_numba_noisy_6q_8l",
            "simulator.shots.dynamic_numba_6q_4r_256s",
            "simulator.guard.sweep_amplitude_8rows_128shots_threads2",
            "simulator.guard.sweep_8q_3l_5p_256rows",
        ):
            self.assertIn(key, guard)
            self.assertIs(guard[key].setup, full[key].setup)
        self.assertEqual(
            {case.parameters.get("method") for case in guard.values()} - {None},
            {"statevector", "density_matrix", "unitary", "superop"},
        )
        for key in (
            "simulator.catalog.method_unitary_numba_2q",
            "simulator.sampling.fixed_state_statevector_numba_5q_65536s",
        ):
            self.assertNotIn(key, guard)
            self.assertIn(key, full)

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

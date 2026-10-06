from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from benchmarks._circuit_catalog import CATALOG, dense_initial_state
from benchmarks.bench_circuit_catalog import CASES, prepare_family
from benchmarks.bench_sampling import prepare_sampling


class CircuitCatalogTests(unittest.TestCase):
    def test_all_families_and_size_series_are_registered(self):
        expected = {
            "ghz_chain",
            "graph_state_line",
            "w_state",
            "efficient_su2_ring",
            "vqe_brickwall",
            "real_amplitudes",
            "qft",
            "qft_entangled",
            "qaoa_maxcut",
            "mid_measure_reset",
            "teleport_feedforward",
            "repetition_code",
            "iterative_feedback",
            "gate_coverage_random",
            "micro_cx",
            "micro_cp",
            "micro_ccx",
            "micro_rz",
            "micro_swap",
            "qutrit_ghz",
            "qutrit_hea",
            "small_unitary",
            "small_superop",
            "initial_state_shallow",
            "initial_state_deep",
            "method_statevector",
            "method_density_matrix",
            "method_unitary",
            "method_superop",
        }
        self.assertEqual({f.name for f in CATALOG}, expected)
        for family in CATALOG:
            for runtime in ("numpy", "numba"):
                cases = [
                    c
                    for c in CASES
                    if c.parameters["family"] == family.name
                    and c.parameters["runtime"] == runtime
                ]
                self.assertEqual(len(cases), len(family.sizes))
                self.assertEqual(
                    [
                        c.parameters.get("qubits", c.parameters.get("qudits"))
                        for c in cases
                    ],
                    list(family.sizes),
                )
            quick = [
                c
                for c in CASES
                if c.parameters["family"] == family.name and "quick" in c.profiles
            ]
            self.assertEqual(len(quick), 1)

    def test_every_family_executes_at_its_smallest_width(self):
        for family in CATALOG:
            with self.subTest(family=family.name):
                prepared = prepare_family(family, family.sizes[0], "numpy")
                result = prepared.run()
                diagnostics = prepared.diagnostics(result)
                if family.export_only:
                    state = getattr(result, f"get_{family.method}")()
                    self.assertTrue(np.isfinite(state).all())
                    if family.method == "statevector":
                        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0)
                else:
                    self.assertEqual(diagnostics["shots_observed"], family.shots)

    def test_method_probes_have_matching_physics(self):
        outputs = {}
        for family in CATALOG:
            if family.name.startswith("method_"):
                result = prepare_family(family, 3, "numpy").run()
                outputs[family.method] = getattr(result, f"get_{family.method}")()
        u, psi = outputs["unitary"], outputs["statevector"]
        np.testing.assert_allclose(u.conj().T @ u, np.eye(8), atol=1e-12)
        np.testing.assert_allclose(psi, u[:, 0], atol=1e-12)
        np.testing.assert_allclose(
            outputs["density_matrix"], np.outer(psi, psi.conj()), atol=1e-12
        )
        np.testing.assert_allclose(outputs["superop"], np.kron(u.conj(), u), atol=1e-12)

    def test_ghz_and_w_counts_have_expected_support(self):
        families = {f.name: f for f in CATALOG}
        ghz = prepare_family(families["ghz_chain"], 5, "numpy").run().get_counts()
        self.assertEqual(set(ghz), {"00000", "11111"})
        w = prepare_family(families["w_state"], 5, "numpy").run().get_counts()
        self.assertEqual(len(w), 5)
        self.assertTrue(all(key.count("1") == 1 for key in w))

    def test_odd_dynamic_layouts_reject_unused_qubits(self):
        for family in CATALOG:
            if family.name in {"teleport_feedforward", "repetition_code"}:
                with self.assertRaisesRegex(ValueError, "odd size"):
                    family.build_fq(4)

    def test_fixed_state_counts_follow_input_probabilities(self):
        shots = 65536
        expected = np.abs(dense_initial_state(5)) ** 2
        prepared = prepare_sampling(5, shots, "numpy")
        counts = prepared.run().get_counts()
        observed = np.array([counts.get(format(i, "05b"), 0) for i in range(32)])
        self.assertEqual(int(observed.sum()), shots)
        # Predeclared six-sigma binomial screen on a deterministic seed.
        tolerance = 6 * np.sqrt(shots * expected * (1 - expected)) + 1
        self.assertTrue(np.all(np.abs(observed - shots * expected) <= tolerance))

    def test_fixed_input_and_backend_are_reused_and_only_counts_requested(self):
        with mock.patch(
            "benchmarks.bench_sampling.fq.simulator.Simulator"
        ) as simulator:
            prepared = prepare_sampling(5, 1024, "numpy")
            prepared.run()
            prepared.run()
        simulator.assert_called_once()
        first, second = simulator.return_value.run.call_args_list
        self.assertIs(first.args[0], second.args[0])
        self.assertIs(first.kwargs["initial_state"], second.kwargs["initial_state"])
        self.assertEqual(
            first.kwargs["result_config"], {"counts": True, "final_state": False}
        )
        np.testing.assert_array_equal(
            first.kwargs["initial_state"], dense_initial_state(5)
        )

    def test_channel_sampling_executes_both_methods(self):
        for kind in ("terminal_noise", "dynamic_noise"):
            for method in ("statevector", "density_matrix"):
                with self.subTest(kind=kind, method=method):
                    prepared = prepare_sampling(
                        3, 32, "numpy", kind=kind, method=method
                    )
                    self.assertEqual(
                        prepared.diagnostics(prepared.run())["shots_observed"], 32
                    )


if __name__ == "__main__":
    unittest.main()

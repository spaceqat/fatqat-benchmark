"""Check fixed-state expectation workloads against small dense operators."""

import unittest
from unittest import mock

from fatqat.observable import NumberOperator, Projector
import numpy as np

import runner
from benchmarks._harness import _selected_cases
from benchmarks.bench_expectation import expectation_inputs, prepare_expectation
from benchmarks.micro.bench_expectation import prepare_contraction


def dense_expectation(state, observable, dimensions):
    paulis = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    total = 0j
    for coefficient, factors in observable.terms:
        local = [np.eye(dim) for dim in dimensions]
        for axis, factor in factors:
            if isinstance(factor, Projector):
                local[axis] = np.zeros((dimensions[axis], dimensions[axis]))
                local[axis][factor.level, factor.level] = 1
            elif isinstance(factor, NumberOperator):
                local[axis] = np.diag(np.arange(dimensions[axis]))
            else:
                local[axis] = paulis[str(factor)]
        matrix = np.ones((1, 1))
        for factor in local:  # public initial states use subsystem order
            matrix = np.kron(matrix, factor)
        total += coefficient * np.vdot(state, matrix @ state)
    return total.real


class ExpectationBenchmarkTests(unittest.TestCase):
    def test_public_and_contraction_match_dense_and_preserve_inputs(self):
        for kind, dimensions, count in (
            ("pauli_sum", (2,) * 4, 8),
            ("projector_sum", (2,) * 4, 32),
            ("mixed_local", (3, 2, 2), 1),
            ("mixed_global", (3, 2, 2), 1),
            ("mixed_projector", (3, 2, 2), 1),
        ):
            with self.subTest(kind=kind):
                state, observable = expectation_inputs(dimensions, kind, count)
                original = state.copy()
                expected = dense_expectation(state, observable, dimensions)
                for module, prepare in (
                    ("benchmarks.bench_expectation", prepare_expectation),
                    ("benchmarks.micro.bench_expectation", prepare_contraction),
                ):
                    with mock.patch(
                        module + ".expectation_inputs", return_value=(state, observable)
                    ):
                        prepared = prepare(dimensions, kind, count)
                    for _ in range(2):
                        result = prepared.run()
                        value = prepared.diagnostics(result)["expectation"]
                        self.assertAlmostEqual(value, expected, places=12)
                    np.testing.assert_array_equal(state, original)

    def test_term_counts_and_repeat_structure(self):
        state, observable = expectation_inputs((2,) * 16, "pauli_sum", 32)
        self.assertEqual(len(observable.terms), 32)
        self.assertEqual(len(set(observable.terms)), 32)
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1)
        repeated, projector = expectation_inputs((2,) * 8, "projector_sum", 256)
        self.assertEqual(len(projector.terms), 256)
        self.assertEqual(len(set(projector.terms)), 8)
        again, _ = expectation_inputs((2,) * 8, "projector_sum", 256)
        np.testing.assert_array_equal(repeated, again)

    def test_guard_keeps_stress_and_private_cases_in_full(self):
        guard = {c.key for c in _selected_cases(runner.CASES, "guard", False)}
        full = {c.key for c in _selected_cases(runner.CASES, "full", False)}
        for name in (
            "exact_numba_12q_16terms",
            "sampled_numba_8q_4terms_256s",
            "fixed_pauli_numba_16q_32terms",
            "fixed_mixed_local_numba_1t14q",
        ):
            self.assertIn("simulator.estimator." + name, guard)
        for name in (
            "fixed_projector_numba_8q_256terms",
            "fixed_mixed_global_numba_1t14q",
            "fixed_mixed_projector_numba_1t19q",
        ):
            self.assertIn("simulator.estimator." + name, full)
            self.assertNotIn("simulator.estimator." + name, guard)
        private = _selected_cases(runner.CASES, "full", True, ["micro.expectation.*"])
        self.assertEqual(len(private), 3)
        self.assertTrue(all(c.unstable and c.key not in full for c in private))

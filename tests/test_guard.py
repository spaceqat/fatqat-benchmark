"""Check guard physics and parameter binding outside timed measurements."""

import unittest

import fatqat as fq
import numpy as np

from benchmarks.bench_guard import (
    prepare_state,
    prepare_sweep,
    structure_sweep_inputs,
    sweep_inputs,
)
from benchmarks.bench_simulator_shots import _parallel_setup


class GuardTests(unittest.TestCase):
    def test_structure_changing_sweep_matches_individually_bound_numpy(self):
        program, theta, values, implementation = structure_sweep_inputs(8, 6)
        backend = fq.simulator.Simulator(
            runtime="numpy", implementation_map=implementation
        )
        options = {
            "shots": 0,
            "simulation_config": {
                "shot_parallelism": "serial",
                "kernel_parallelism": "serial",
            },
            "result_config": {"counts": False, "final_state": True},
        }
        prepared = prepare_sweep(rows=6, structure_changing=True)
        results = prepared.run()
        self.assertEqual(prepared.diagnostics(results)["rows"], 6)
        for value, result in zip(values, results):
            expected = (
                backend.run(program.assign_parameters({theta: value}), **options)
                .result()
                .get_statevector()
            )
            np.testing.assert_allclose(result.get_statevector(), expected, atol=1e-12)
        for index in (1, 2):
            self.assertFalse(
                np.allclose(
                    results[0].get_statevector(), results[index].get_statevector()
                )
            )

    def test_chunk_boundary_rows_match_individual_bindings(self):
        prepared = prepare_sweep(
            width=13, layers=1, rows=2049, kernel_mode="auto", workers=4
        )
        results = prepared.run()
        self.assertEqual(prepared.diagnostics(results)["rows"], 2049)
        program, angles, values = sweep_inputs(13, 1, 2049, 5)
        backend = fq.simulator.Simulator(runtime="numpy")
        for index in (0, 2047, 2048):
            expected = (
                backend.run(
                    program.assign_parameters({angles: values[index]}),
                    shots=0,
                    simulation_config={
                        "shot_parallelism": "serial",
                        "kernel_parallelism": "serial",
                    },
                    result_config={"counts": False, "final_state": True},
                )
                .result()
                .get_statevector()
            )
            np.testing.assert_allclose(
                results[index].get_statevector(), expected, atol=1e-12
            )

    def test_parameter_sharing_keeps_gate_count_and_matches_individual_runs(self):
        counts = []
        for parameters in (1, 5, 32):
            program, angles, values = sweep_inputs(4, 4, 3, parameters)
            counts.append(len(program.dag().nodes))
            backend = fq.simulator.Simulator(runtime="numpy")
            config = {"shot_parallelism": "serial", "kernel_parallelism": "serial"}
            options = {
                "shots": 0,
                "simulation_config": config,
                "result_config": {"counts": False, "final_state": True},
            }
            batch = backend.run_sweep(program, {angles: values}, **options).result()
            for row, result in zip(values, batch):
                bound = program.assign_parameters({angles: row})
                expected = backend.run(bound, **options).result().get_statevector()
                np.testing.assert_allclose(
                    result.get_statevector(), expected, atol=1e-12
                )
        self.assertEqual(len(set(counts)), 1)

    def test_numba_sweep_and_state_match_numpy(self):
        for prepare, options in (
            (prepare_sweep, {"rows": 3}),
            (prepare_state, {"width": 6, "layers": 3, "ccx": True}),
        ):
            outputs = []
            for runtime in ("numpy", "numba"):
                prepared = prepare(runtime=runtime, **options)
                result = prepared.run()
                prepared.diagnostics(result)
                outputs.append(result if isinstance(result, list) else [result])
            for expected, actual in zip(*outputs):
                np.testing.assert_allclose(
                    actual.get_statevector(), expected.get_statevector(), atol=1e-12
                )

    def test_all_noise_channels_and_shot_thread_workload_execute(self):
        for noise in ("amplitude", "phase", "depolarizing"):
            prepared = prepare_sweep(
                rows=2, layers=1, noise=noise, shots=2, runtime="numpy"
            )
            self.assertEqual(prepared.diagnostics(prepared.run())["rows"], 2)
        # Unlike terminal-only sampling, this recipe has independent trajectories.
        prepared = _parallel_setup(3, 8, "serial")
        self.assertEqual(sum(prepared.run().get_counts().values()), 8)


if __name__ == "__main__":
    unittest.main()

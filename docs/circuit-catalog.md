# Circuit catalog

List every exact case ID, including optional private microbenchmarks and QASM:

```sh
python runner.py --profile full --include-micro --list
python qasm_runner.py --profile full --list
```

The scalable circuit families below live in `benchmarks/bench_circuit_catalog.py`.

## Circuit families

| Section | Families | Output |
| --- | --- | --- |
| A | ghz_chain, graph_state_line, w_state | 1,024 terminal counts |
| B | efficient_su2_ring, vqe_brickwall, real_amplitudes | 1,024 terminal counts |
| C | qft, qft_entangled, qaoa_maxcut | 1,024 terminal counts |
| D | mid_measure_reset, teleport_feedforward, repetition_code, iterative_feedback | 32 dynamic counts |
| E | gate_coverage_random, micro_cx, micro_cp, micro_ccx, micro_rz, micro_swap | 1,024 terminal counts |
| F | qutrit_ghz, qutrit_hea | Statevector |
| G | small_unitary, small_superop (includes reset) | Full operator |
| H | initial_state_shallow, initial_state_deep | Statevector from fixed dense input |
| M | method_statevector, method_density_matrix, method_unitary, method_superop | Same four-layer RY/RZ/CX circuit in each representation |

Full includes static circuits up to 24
qubits, deep circuits up to 20, dynamic circuits up to 20 (odd repeater/code
layouts up to 17), qutrit circuits up to 12, unitaries up to 8 and superoperators
up to 5. Each point runs separately with NumPy and Numba. The quick profile
selects the smallest size of every family with Numba. Larger representations
can be expensive; each case has its own timeout.

## Catalog cases in guard

Guard reuses these existing circuits with Numba only. Each case requests 1,024
terminal counts and uses serial execution, so sizes can be compared without
also changing the threading mode.

| Family | Qubits | Circuit |
| --- | --- | --- |
| `ghz_chain` | 5, 15, 24 | H followed by a CX chain |
| `qft` | 5, 15, 24 | H, controlled phases and swap reversal |
| `qaoa_maxcut` | 5, 15, 24 | Ring ZZ cost and RX mixer |
| `efficient_su2_ring` | 5, 15, 20 | RY/RZ layers with a CX ring |
| `vqe_brickwall` | 5, 15, 20 | RX/RY/RZ layers with CX brickwork |
| `real_amplitudes` | 5, 15, 20 | RY layers with a CX ladder |

The variational layer count is `max(2, qubits // 4)`: 2, 3 and 5 layers at
these sizes. Guard's separate state-evolution cases provide 1/64-layer
coverage. Case IDs use `simulator.catalog.<family>_numba_<qubits>q`.

Guard also reuses these eight Numba cases:

| Case ID | Purpose |
| --- | --- |
| `simulator.shots.dynamic_numba_6q_4r_256s` | SV measurement, reset and conditional gates |
| `simulator.state.dm_numba_noisy_6q_8l` | Noisy DM evolution with final-state output |
| `simulator.state.mixed_qudit_numba` | Mixed qubit/qutrit state evolution |
| `simulator.catalog.method_unitary_numba_8q` | 8-qubit unitary construction |
| `simulator.catalog.small_superop_numba_4q` | 4-qubit superoperator with reset |
| `simulator.sampling.fixed_state_statevector_numba_12q_65536s` | 65,536 samples from a fixed dense SV |
| `simulator.sampling.terminal_noise_density_matrix_numba_8q_32s` | Noisy DM evolution followed by terminal sampling |
| `simulator.sampling.dynamic_noise_density_matrix_numba_5q_32s` | Noisy DM with intermediate measurement and reset |

Guard also includes the eight sweeps below and four state-evolution cases:
16 qubits at 1/64 layers with serial execution, 16 qubits at 64 layers with
automatic threading, and 20 qubits at 3 layers with four kernel workers.
Process-shot and explicit fusion comparisons are in full.

## Parameter sweeps

Each row supplies a new parameter binding to the same circuit. The timed
operation is the complete `run_sweep(...).result()` call, including binding
and simulation. Full has 17 sweep cases; guard selects eight of them.

| Case ID or pattern | Circuit and parameters | Batch rows / shots per row | Profiles |
| --- | --- | --- | --- |
| `simulator.sweep.numba_8q_4l_16rows` | 8 qubits, 4 layers, 8 parameters | 16 / 0 | quick, full |
| `simulator.sweep.numba_10q_8l_64rows` | 10 qubits, 8 layers, 10 parameters | 64 / 0 | full |
| `simulator.guard.sweep_8q_3l_5p_*rows` | 8 qubits, 3 layers, 5 shared parameters | 1, 8, 64, 256 / 0 | full; 8/64/256 rows also in guard |
| `simulator.guard.sweep_8q_16l_5p_64rows` | 8 qubits, 16 layers, 5 parameters | 64 / 0 | guard, full |
| `simulator.guard.sweep_8q_16l_128p_64rows` | Same gate count with 128 parameters | 64 / 0 | full |
| `simulator.guard.sweep_*_8rows_2shots` | 8 qubits, 3 layers, 5 parameters; amplitude damping, phase damping or depolarizing noise | 8 / 2 | guard, full |
| `simulator.guard.sweep_amplitude_*rows_2shots` | Same circuit with amplitude damping | 8, 64, 256 / 2 | full; 8 rows also in guard |
| `simulator.guard.sweep_amplitude_*rows_128shots_threads2` | Same circuit; two shot workers | 8, 64 / 128 | full; 8 rows also in guard |
| `simulator.guard.sweep_13q_1l_5p_2049rows_auto4` | 13 qubits, 1 layer, 5 parameters; crosses the reviewed 2,048-row state-buffer boundary | 2,049 / 0 | full |
| `simulator.guard.sweep_structure_8q_64rows` | 8 qubits, one scalar parameter switching custom gates between diagonal, permutation and dense matrices on unordered targets | 64 / 0 | full |

Patterns overlap for the 8-row amplitude-damping case. All these cases use
Numba; zero shots requests final statevectors. Guard noise probabilities are
0.03. The 2,049-row case returns about 256 MiB of states plus working memory;
internal chunking depends on the FatQat revision.

List all sweep IDs, or run one example:

```sh
python runner.py --profile full --case "*sweep*" --list
python runner.py --profile guard --case "simulator.guard.sweep_8q_3l_5p_64rows"
```

Definitions are in `benchmarks/bench_sweep_estimator.py` and
`benchmarks/bench_guard.py`.

## Additional sampling coverage

`fixed_state` allocates a seeded normalized dense statevector during setup and
reuses it with a measurement-only program. Widths are 5, 12 and 20 qubits, with
1,024 and 65,536 shots per width and runtime. Both shot counts at 5 qubits with
Numba are in quick. The timed `Simulator.run(...).result()` call includes input
validation, copying, sampling and result assembly, with no gate evolution.

`terminal_noise` applies amplitude damping after RY gates in an SU2 ring.
`dynamic_noise` adds depolarizing noise after CX gates to the mid-measure/reset
family. Both request 32 counts, use statevector and density-matrix methods, and
cover widths 5 and 8 with both runtimes. The 5-qubit Numba cases are in quick.
These cover trajectory sampling, mixed-state sampling, and measurement/reset
with channel noise. All counts cases set `final_state=False` explicitly.

## Other workload groups

| Case group | Coverage | Definitions |
| --- | --- | --- |
| `simulator.state` | NumPy/Numba statevectors, noisy density matrices, explicit DM fusion off/on, unitary/superoperator output, mixed qubits/qutrits | `bench_simulator_state.py` |
| `simulator.shots` | Measurement and conditional gates: 6 qubits / 256 shots and 8 qubits / 1,024 shots | `bench_simulator_shots.py` |
| `simulator.parallel` | Same 6-qubit dynamic circuit with serial, thread and process shot execution | `bench_simulator_shots.py` |
| `simulator.guard` (non-sweep) | 16-qubit shallow/deep circuits, auto threading, 20-qubit threaded execution; local CCX with eight workers is full-only | `bench_guard.py` |
| `simulator.estimator` | Exact expectation values at 8/12 qubits and sampled estimates at 8 qubits / 256 shots | `bench_sweep_estimator.py` |
| `compiler.sc` | Local line and nonlocal grid routing at 12–32 qubits | `bench_compile_sc.py` |
| `compiler.na` | Round-robin interactions on 20/50 atoms | `bench_compile_na.py` |
| `micro.*` | Private gate, channel, SABRE routing and ZAP scheduling diagnostics; opt in with `--include-micro` | `micro/bench_*.py` |
| `qasm.*` | QASM 2/3 import, import plus simulation, and SC/NA compilation; 16 cases, 8 in quick | `bench_qasm_*.py` |

Definition paths are relative to `benchmarks/`. The Python SC/NA compiler cases
require older APIs; see [compatibility notes](running-benchmarks.md#compatibility).
QASM has quick/full profiles only. Some `simulator.guard.*` cases are full-only;
the IDs stay unchanged for existing result files, but profiles control selection.

## Seeds and timing

Simulator programs, backends, noise models and dense inputs are constructed in
`setup_ns`; timed simulator calls include execution and result retrieval.
The scalable catalog and sampling cases use serial shot/kernel execution and
seed 7. The cases in `bench_guard.py` use seed 17 and include threaded cases;
cases reused by guard retain their original settings. Compiler, QASM and private
microbenchmarks have their own timing boundaries.
See the [running guide](running-benchmarks.md) for measurement details and checks.

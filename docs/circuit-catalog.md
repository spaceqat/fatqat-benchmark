# Circuit catalog

The scalable circuit catalog is implemented by
`benchmarks/bench_circuit_catalog.py`; `benchmarks/bench_sampling.py` adds
counts-only sampling cases.

## Coverage

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
| M | method_statevector, method_density_matrix, method_unitary, method_superop | Matching four-layer RY/RZ/CX probe in each representation |

The full profile covers the full size series: static circuits up to 24
qubits, deep circuits up to 20, dynamic circuits up to 20 (odd repeater/code
layouts up to 17), qutrit circuits up to 12, unitaries up to 8 and superoperators
up to 5. Each point runs separately with NumPy and Numba. The quick profile
selects the smallest size of every family with Numba. Larger representations
can be expensive; each case retains the harness's independent timeout.

The catalog follows this repository's Python-native benchmark scope: it compares
no other simulators and requires no new runtime dependency. Existing QASM cases
and their timing boundaries remain separate.

## Additional sampling coverage

`fixed_state` allocates a seeded normalized dense statevector during setup and
reuses it with a measurement-only program. Widths are 5, 12 and 20 qubits, with
1,024 and 65,536 shots per width and runtime. Both shot counts at 5 qubits with
Numba are in quick. This isolates the public counts-request workflow from gate
evolution; input validation/copying, sampling and result assembly still belong
to the measured `Simulator.run(...).result()` call. It is not a private random
number generator microbenchmark.

`terminal_noise` applies amplitude damping after RY gates in an SU2 ring.
`dynamic_noise` adds depolarizing noise after CX gates to the mid-measure/reset
family. Both request 32 counts, use statevector and density-matrix methods, and
cover widths 5 and 8 with both runtimes. The 5-qubit Numba cases are in quick.
These cover trajectory sampling, mixed-state sampling, and measurement/reset
with channel noise. All counts cases set `final_state=False` explicitly.

## Measurement and validation

Programs, backends, noise models and dense inputs are constructed in `setup_ns`.
Each timed operation uses the complete public simulator portal. Shot and kernel
parallelism are fixed to serial, and the RNG seed is fixed at 7. Method, runtime,
width, shot count and output mode are recorded per case.
Counts diagnostics check the requested total after timing. State diagnostics
record output shape and bytes. Infrastructure regression tests exercise all
families at small widths, method agreement, GHZ/W support, fixed-state sampling
probabilities, input reuse and noisy execution. They supplement, rather than
replace, FatQat's correctness suite.

```sh
python benchmarks/bench_circuit_catalog.py --list
python benchmarks/bench_circuit_catalog.py --profile quick
python benchmarks/bench_sampling.py --profile quick
python runner.py --profile full --list
python -m unittest discover -s tests -v
```

The main suite has 45 quick cases and 302 full cases, excluding optional
microbenchmarks. The catalog contributes 29 quick / 246 full cases and the
additional sampling module contributes 6 quick / 28 full cases.

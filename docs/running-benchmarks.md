# Running benchmarks

Start with the profiles and commands in the [README](../README.md).

## Select cases and check a local PR

The architecture has three small parts: **case files** build a workload and
provide its timed operation; the **harness** selects cases and measures them
in isolated workers; **compare.py** runs the same selection against two installed
FatQat revisions. Python-native and QASM tracks keep their existing boundaries.
There is no environment manager or automatic Git checkout switching.

Use `quick` for the existing small catalog, `full` for all sizes, or `guard`
for 38 representative simulator cases. Guard includes GHZ, QFT and QAOA at
5/15/24 qubits, and efficient SU2, VQE brickwall and real-amplitude circuits at
5/15/20 qubits, all with Numba. It also covers 8/64/256-row parameter batches,
shallow/deep circuits, three noise channels and kernel/shot threading, plus
existing dynamic, DM, qudit, 8-qubit unitary and 4-qubit superoperator cases.
Sampling includes a fixed 12-qubit state and both terminal and dynamic noisy
DM workloads. The largest states make guard slower than a small-circuit smoke
test; use `--case` to narrow a local investigation.

Full-only probes include a 13-qubit, 2,049-row workload (crossing the
reviewed 256 MiB batch-state limit at 2,048 rows), amplitude-damping batches
of 64/256 rows with two shots and 64 rows with 128 shots, and a custom gate
that switches between diagonal, permutation and dense matrices on unordered
targets. The 128-parameter sweep and local CCX/eight-worker regression are
also full-only, along with the 1-row noiseless sweep variant.
Case IDs retain their original `simulator.guard.*` prefix; profile membership
controls selection. The boundary workload returns about 256 MiB of states,
with additional working memory; its internal chunking depends on the FatQat revision.

```sh
python runner.py --profile guard --list
python runner.py --profile full --case "simulator.catalog.qft_*" --list
python runner.py --profile guard --case "*sweep*" --exclude "*128shots*"
```

`--case` and `--exclude` are repeatable, case-sensitive globs over the printed
case IDs. Includes are combined with OR, then exclusions apply; the profile
still limits selection. Empty selections fail. `--describe` prints the selected
parameters and environment as JSON without running workloads. Individual
benchmark files and `qasm_runner.py` support the same filters.

To run one benchmark file or the QASM track:

```sh
python benchmarks/bench_sampling.py --profile quick
python qasm_runner.py --profile quick
```

Use `--warmups`, `--repeats` and `--timeout` to adjust measurement settings.
Optional microbenchmarks use private FatQat APIs and may break between versions;
enable them with `--include-micro` on the main runner.

Ordinary runs print a summary and save JSON under `results/` (or `--output`).
The file includes raw timings, environment metadata, diagnostics, failures and
timeouts. Each result records its actual repeat count; some cases use fewer
repeats than the profile default.

Install base and candidate FatQat in separate environments with matching Python
and dependency versions, then pass their Python executable paths:

```sh
python compare.py --base-python BASE_PYTHON --candidate-python PR_PYTHON --case "*sweep*" --output results/pr-sweeps.json
```

Replace `BASE_PYTHON` and `PR_PYTHON` with the executable paths from the respective
environments. Run `python -c "import sys; print(sys.executable)"` in each
environment to find its path; quote paths containing spaces.

Comparison defaults to `guard`; use `--runner qasm_runner.py --profile quick`
for QASM. Workers run sequentially in seeded randomized base/candidate order
for five paired rounds.
Each worker warms up and averages complete calls over at least 50 ms per sample.
Disk compilation caches are separate per revision and reused across rounds;
first-call timings are recorded but are **not** a fresh-cache JIT comparison.
Completed pairs are saved as the comparison progresses; an interrupted report
keeps `complete: false` and must not be treated as a passing run.

HTML shows per-case times, verdicts, source hashes, environment details and an
estimated 95% paired bootstrap interval; its linked JSON also contains raw
samples and worker diagnostics. A slowdown
above `--threshold 0.05` (5%) is flagged only when the interval clears the budget;
an interval crossing it is inconclusive. `--min-change-ms` can add an absolute
budget per call (default zero). Confirm flagged cases with `--rounds 11
--min-sample-time 0.2`; do not treat one run as proof across all workloads.
Existing output files are never overwritten by the comparison command.

Exit codes: **0** all selected cases pass, **1** regression, **2** error or
incompatible environments, **3** inconclusive. Failures and unsupported modes
remain visible; they are not performance wins. Run correctness tests separately.
Guard diagnostics check state norms and shot totals; its tests also compare
representative Numba outputs with NumPy and sweeps with individual bindings.

Use an idle machine, the same CPU affinity and BLAS thread limits for both
environments. The command inherits and records thread settings rather than
changing machine policy. Explicit threaded cases require a capable Numba build;
exclude them on unsupported machines. The legacy terminal-GHZ shot-scaling
cases were replaced with newly named dynamic cases because terminal sampling
does not require independent per-shot evolution.

## Compatibility

The legacy Python SC/NA compiler cases still require older FatQat compiler
APIs; they are outside `guard` and can fail on newer installations.

## Measurement model

- Workload construction is outside the primary timed operation and is reported
  separately as `setup_ns`.
- `cold_ns` measures the first operation in a worker. Ordinary runs use a fresh
  Numba cache per case; comparisons reuse separate per-revision caches across
  rounds. Neither timing includes interpreter startup or module imports.
- Warm measurements follow untimed warm-up calls and use the median as the
  primary summary.
- Each case runs in its own subprocess. A timeout is recorded without stopping
  the remaining cases.
- The primary baseline fixes simulator shot and kernel parallelism to serial.
  Parallel scaling cases are separate.
- Unix workers report process peak RSS with `resource.getrusage`; unsupported
  platforms report `null`. Peak RSS is platform-specific and must not be
  compared across operating systems.
- Simulator, SC compiler, and NA compiler results remain separate. The runner
  deliberately does not emit a single aggregate score.

The suite requires Python 3.12 or newer and otherwise uses only the standard
library plus FatQat and its installed dependencies.
When the optional `git` executable is unavailable, Git commit and dirty-state
metadata are reported as `null` rather than preventing benchmark output.

## QASM corpus

QASM files and their provenance live under `corpora/qasm/`. The TOML manifest
records each stable fixture ID, format, profile, SHA-256 digest, originating
workload, exporter, FatQat revision, license, and paired Python-native case.
Manifest parsing, file reads, and digest verification happen during case setup,
outside the primary timed operation. Benchmark runs never generate or download
QASM input.

## Validate the benchmark infrastructure

Run the standard-library regression suite before accepting benchmark changes:

```sh
python -m unittest discover -s tests -v
```

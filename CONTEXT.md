# Benchmark Evaluation

This context defines how FatQat performance workloads are named and separated so
that results with different input and execution boundaries are not conflated.

## Language

**Python-native benchmark case**:
A workload authored through FatQat's Python interface before its timed operation
begins.
_Avoid_: Python benchmark, native benchmark

**QASM import benchmark**:
A workload that measures the transformation of in-memory OpenQASM source into a
FatQat Program.
_Avoid_: Parse-only benchmark, parser benchmark

**QASM compilation benchmark**:
A workload that measures OpenQASM source through a target compiler and excludes
execution of the compiled result.
_Avoid_: QASM compiler benchmark, compile benchmark

**QASM execution benchmark**:
A workload that measures OpenQASM import followed by simulation of the resulting
FatQat Program.
_Avoid_: QASM simulation benchmark, simulation benchmark

**Performance corpus**:
The fixed set of supported inputs used for timing and resource comparisons.
_Avoid_: Compatibility suite, test corpus

**Compatibility corpus**:
Inputs used to characterize accepted and rejected language features without
contributing to performance results.
_Avoid_: Failure benchmark, performance corpus

**Paired workload**:
Semantically equivalent Python-native and QASM inputs that expose the additional
cost of a text frontend without merging their results.
_Avoid_: Duplicate benchmark

**Corpus manifest**:
The provenance and identity record for a versioned benchmark input, including
its immutable digest and license.
_Avoid_: File list, benchmark configuration

**Materialized QASM fixture**:
OpenQASM source produced offline by a named exporter and committed as an
immutable corpus input for benchmark runs.
_Avoid_: Generated benchmark, runtime-generated circuit

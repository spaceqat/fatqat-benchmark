---
status: accepted
---

# Separate Python-native and QASM benchmark tracks

FatQat Benchmark preserves Python-native and QASM workloads as separate tracks
and runners because their timed boundaries differ: QASM import constructs a
Program, while QASM compilation and execution include additional frontend work.
Keeping the tracks separate protects the established Python-native case set,
runtime, and memory baseline while still allowing semantically paired workloads;
stage cost is observed directly and is not inferred by subtracting medians from
different benchmark cases.

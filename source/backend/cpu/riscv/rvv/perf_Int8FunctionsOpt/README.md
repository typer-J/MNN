# RVV Int8FunctionsOpt function-level perf tests

Each optimized int8 function has one standalone benchmark source named `test_perf_<function>.cpp`.
The RVV implementation included by each benchmark is the file in the parent demo directory, for example `../MNNFloat2Int8.cpp`.

The binary argument selects the implementation:

- `0`: scalar reference mode
- `1`: RVV optimized mode

Run all functions on the RISC-V machine:

```bash
cd /data/xjj/Int8FunctionsOpt/perf_Int8FunctionsOpt
chmod +x run_perf_all.sh
./run_perf_all.sh
```

Run one function:

```bash
./run_perf_all.sh MNNFloat2Int8
```

Use a fixed CPU or different perf events:

```bash
TASKSET_CPU=0 REPEAT=10 EVENTS=cycles,instructions,branches,branch-misses ./run_perf_all.sh MNNFloat2Int8
```

The input data is generated from fixed seeds inside each benchmark, so scalar and RVV modes use identical test cases.
The result workbook is intentionally not included yet.

# RVV function-level perf tests

Each optimized function has one standalone benchmark source named `test_perf_<function>.cpp`.
The RVV implementation included by each benchmark is the file in the parent demo directory, for example `../MNNAbsMaxFP32.cpp`.

The binary argument selects the implementation:

- `0`: scalar reference mode
- `1`: RVV optimized mode

Run all functions on the RISC-V machine:

```bash
cd /data/xjj/CommonOptFunction/demo/perf
chmod +x run_perf_all.sh
./run_perf_all.sh
```

Run one function:

```bash
./run_perf_all.sh MNNAbsMaxFP32
```

Use a fixed CPU or different perf events:

```bash
TASKSET_CPU=0 REPEAT=10 EVENTS=cycles,instructions,branches,branch-misses ./run_perf_all.sh MNNAbsMaxFP32
```

The input data is generated from fixed seeds inside each benchmark, so scalar and RVV modes use identical, uniformly distributed test cases.

`MNNAsyQuantInfo_FP32.cpp` now has a canonical RVV implementation, so it participates in `run_perf_all.sh` like the other targets.

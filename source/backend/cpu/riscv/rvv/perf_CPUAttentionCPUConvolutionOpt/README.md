# RVV CPUAttention / CPUConvolution perf tests

Each optimized function has one standalone benchmark source named `test_perf_<function>.cpp`.
The perf directory is expected to be placed at:

```bash
/data/xjj/CPUAttentionCPUConvolution/perf_CPUAttentionCPUConvolutionOpt/
```

The RVV implementation files are expected to be placed in the parent directory:

```bash
/data/xjj/CPUAttentionCPUConvolution/
```

Each benchmark includes its RVV implementation with a relative path, for example:

```cpp
#include "../MNNAttentionMaskQK.cpp"
```

The binary argument selects the implementation:

- `0`: scalar reference mode
- `1`: RVV optimized mode

Run all functions on the RISC-V machine:

```bash
cd /data/xjj/CPUAttentionCPUConvolution/perf_CPUAttentionCPUConvolutionOpt
chmod +x run_perf_all.sh
./run_perf_all.sh
```

Run one function:

```bash
./run_perf_all.sh MNNAttentionMaskQK
```

Use a fixed CPU or different perf events:

```bash
TASKSET_CPU=0 REPEAT=10 EVENTS=cycles,instructions,branches,branch-misses ./run_perf_all.sh MNNAttentionMaskQK
```

The input data is generated from fixed seeds inside each benchmark, so scalar and RVV modes use identical test cases.

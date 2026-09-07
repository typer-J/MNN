# MNN scalar / RVV C4 microbenchmark

Compares the four production RVV functions `MNNPackC4_RVV`, `MNNUnpackC4_RVV`,
`MNNScaleAndAddBias_RVV`, and `MNNReluWithSlopeChannel_RVV` with MNN's current
generic implementations in the same executable. This is a kernel microbenchmark;
it does not establish model-level speedups, CPU dispatch coverage, or multithreaded
Execution correctness.

## Baseline provenance

`run.py` snapshots the four RVV translation units, their private header, the
generic `CommonOptFunction.cpp`, and the real `math/Vec.hpp` dependencies. It
extracts the exact six relevant definitions (four functions plus two pack
templates) from the generic source without rewriting their bodies. Scale uses
MNN's actual `Vec<float, 4>` implementation. The independent elementwise/layout
specification in the harness is a correctness check, **not** the timed baseline.

The scalar translation unit uses `-O3 -fno-tree-vectorize
-fno-tree-slp-vectorize -march=rv64gc` with GCC (the explicit base ISA also
prevents vector lowering of aggregate copies). Label its measurements
`mnn_scalar_no_autovec`; this intentionally excludes compiler vectorization and
is not a claim to beat an unrestricted `-O3` generic baseline. Both variants use
`-fno-fast-math -ffp-contract=off`, with LTO disabled. No compiler version is
assumed from a directory name: the runner records version, target triple,
invoked compiler driver/wrapper hash, complete sanitized compiler arguments,
flags, source/fragment hashes and object hashes. Git HEAD and the scoped file
status are recorded when Git metadata is available. With a compiler wrapper,
the invoked executable hash identifies that wrapper, not the inner compiler.

## Native Linux RISC-V run (GCC 14 supported by command interface)

Run from a checkout containing the candidate functions. These are examples,
not evidence that hardware testing has already passed:

```bash
CXX=g++ python3 benchmark/rvv_scalar_compare/run.py --out /path/to/new-result
```

The runner compiles with `-march=rv64gcv -mabi=lp64d`, executes a real RVV FP32
load/add/store probe and measures VLEN. It does not assume that the CPU is an
SG2044. Its tier is `target_riscv64_execution`: the caller must independently
establish whether this is physical hardware or a RISC-V emulator, because
`uname` and successful instructions alone cannot distinguish them. The probe
also records `riscv_hwprobe` status and its all-online-CPU V capability bit.
An executable RVV instruction does not imply that MNN's runtime enables RVV
(for example, older kernels can lack hwprobe); direct kernel invocation here
does not validate the full MNN function-table dispatch. If the intended target's VLEN and CPU set have already been established,
require them explicitly (VLEN is in **bits**):

```bash
EXPECT_VLEN=<verified-vlen-bits> CXX=<compiler> python3 benchmark/rvv_scalar_compare/run.py \
  --cpus <logical-cpu-id> --out /path/to/new-result
```

CPU affinity applies to both the probe and benchmark and is recorded. Without
`--cpus`, inherited affinity is recorded; this does not prove a particular
microarchitecture/cluster was exercised. No remote username, hostname, IP,
SSH credentials or board label is required or recorded. Use fresh output
directories and new processes for repeated runs. Record governor, frequency,
temperature and competing load alongside the result when comparing runs.
`--cpu` is an alias for `--cpus`; `CPUSET` supplies the same comma-separated
IDs through the environment. `--correctness-only` still runs the target probe
but omits all timing. Host checks with the current MSVC standard library use
C++14 and record that exception; the target harness remains C++11.

## Cases and timing

- Areas: `0,1,3,4,7,16,257,1024`, plus measured e32m8 VL boundaries
  `VL-1,VL,VL+1,2*VL+1`.
- Channels: `0,1,3,4,5,7,8,9,13,32,64`. Elementwise APIs receive the rounded
  C4 group count, which is also recorded.
- Pack/unpack: contiguous and two asymmetric stride combinations; padding and
  prefix/suffix guards must remain intact. Pack must zero unused C4 lanes.
- Scale/PReLU: out-of-place and exact `dst == src`; random positive/negative
  data and parameters plus signed zero, subnormal, Inf and NaN boundaries.
  Pack/unpack compare bits; elementwise results compare bits except NaNs are
  compared by class. Fast math and FMA contraction are disabled for these checks.
- Both scalar and RVV results must independently satisfy the layout/elementwise
  contract. Correctness failures are retained and return a nonzero exit; the
  failed case is not timed, and remaining cases still execute.
- Four warmup pairs and adaptive calibration are excluded from measurements.
  Both implementations calibrate, and the faster side must reach the requested
  batch duration before sampling (subject to a 2^22 iteration cap).
  Each round alternates `AB/BA` order inside one binary; every sample retains
  iterations, total nanoseconds and nanoseconds per call. Default: seven rounds,
  approximately 1 ms scalar batches (`--rounds`, `--sample-ms`).
- Timed in-place Scale uses `alpha=-1,bias=0` and PReLU uses slope 1 to prevent
  repeated application from drifting into infinity/subnormal behavior. Their
  correctness tests still use random and boundary parameters. This special
  timing input is explicitly recorded, so it must not be confused with random
  in-place workload performance.

Outputs: `metadata.json`, exact `source_snapshot/`, generated `scalar.cpp`,
compiler/run logs, `results.jsonl` (correctness and raw samples), `samples.csv`,
`comparison.csv` (per-case medians, minimum/maximum, MAD and scalar/RVV ratio),
objects and the executable. Empty-work cases retain their timings but have no
reported speedup. No cross-case aggregate speedup is produced. A failure retains its directory. The runner checks
source hashes immediately before execution and after collection and rejects a
cohort if production or harness source changed during the run. An archive
without `.git` works; source hashes are the source identity.

## Host-only harness validation

```bash
CXX=clang++ python3 benchmark/rvv_scalar_compare/run.py --host-scalar-check \
  --out /path/to/new-host-check
```

This compiles exact MNN scalar functions and a deliberately labelled scalar
alias to test extraction, independent contract checks and result handling.
It runs correctness only and records `host_scalar_harness_check`; it produces
**no RVV hardware or speedup evidence**. `--extract-only` validates extraction
without a compiler. A separate semantic-emulation driver can compile the
production RVV source with an intrinsic shim and invoke the executable with
`--correctness-only --vlen-bits <emulated-bits> --evidence-tier host_semantic_emulation`; label that result as emulation,
not hardware execution.

## SG2044 validation entrypoint

`sg2044_validate.sh` orchestrates GCC 14 preflight, hardware/affinity recording,
the correctness run, three fresh performance processes, and `integration.sh`
for separate full MNN scalar/RVV builds. Run it from a fresh isolated checkout;
it requires `ORIGINAL_REPO` and refuses to write inside that existing checkout.
Pass site paths through the environment:

```bash
ORIGINAL_REPO=<existing-mnn-checkout> GCC14_PREFIX=<gcc14-install-prefix> \
  bash benchmark/rvv_scalar_compare/sg2044_validate.sh
```

The default VLEN expectation is 512 bits. The target probe must confirm it;
the default is an expectation, not measured hardware identity. Set `EXPECT_VLEN`
only to an independently established value. `CPUSET` selects logical CPU IDs;
without it the script selects an allowed CPU and records that choice. Pinning
does not reserve a core or eliminate interference on a shared server.

`PERF_PROCESSES` controls fresh benchmark processes (default 3). `JOBS` controls
integration build parallelism (default 4). `SKIP_INTEGRATION=1` explicitly skips
the full-library stage and is recorded as incomplete integration coverage.
`CC`/`CXX` can name explicit executable paths if the installation does not use
the usual `bin/gcc` and `bin/g++` layout. No toolchain or debugger is installed.

The integration stage checks real test process exit codes and nonempty passing
summaries for both variants. Its small op cases check integration correctness;
they do not establish large-tensor scaling or model throughput. When an existing
GDB is available, a separate untimed run checks the four RVV entrypoints. Model
smoke tests still require representative local models and remain a separate
validation stage. Review the raw results and per-case variance before accepting
any speedup; script completion alone is not a performance acceptance decision.

---
name: test-ci
description: Run MNN tests / benchmarks on host or real devices. Covers two parallel tracks — (1) the regression / CI suite (static checks, host-side tests, on-device Android arm64 matrix via ./test.sh + test_stages.json) and (2) one-command iOS real-device LLM benchmarking (prefill/decode tok/s, branch comparison). Use when the user asks to run the tests, run CI, smoke-test a build, verify a change on a device, benchmark on-device (Android or iPhone/iPad), or add / select / retune a test stage.
---

# MNN Test / CI SKILL (index)

This skill is an **index**. Pick the document that matches the task and follow it:

| Document | Use when |
|----------|----------|
| [`test-suite.md`](test-suite.md) | Run the regression / CI suite — static checks, host (local) tests, the on-device **Android** arm64 matrix (`./test.sh` + `test_stages.json`); add / select / retune a test stage; audit stale CI scripts; add a new operator test. |
| [`ios-llm-bench.md`](ios-llm-bench.md) | Benchmark LLM prefill/decode speed on a real **iPhone/iPad** (`ios_llm_bench.sh`); compare branches on iOS Metal/CPU; verify Metal kernel changes on device. |

The two tracks are independent: Android/host regression testing goes through
`test.sh`, while iOS LLM benchmarking goes through
`transformers/llm/engine/ios/ios_llm_bench.sh`.

| File | Role |
|------|------|
| [`test.sh`](../../test.sh) | Bash driver. `static`, `local` (host CPU), and `android <serial>` modes. |
| [`test_stages.json`](../../test_stages.json) | Declarative stage matrix. **Edit this** to add / drop / retune stages — no shell edits needed for the common cases. It is self-documenting via its `_documentation` block. |
| [`docs/testing.md`](../../docs/testing.md) | 中文测试文档：阶段说明、字段表、新增算子测试流程。 |


## Target-revision preflight

When testing a branch in a secondary worktree, verify that `test.sh` and its
configuration files exist in that exact worktree before invoking the driver.
The skill instructions in the primary checkout can be newer than the target
revision. If the target branch does not contain the driver, do not copy newer
test infrastructure into the feature branch solely to run validation. Use the
target revision's native CMake build plus `run_test.out` instead, and report
that the declarative driver was unavailable on that revision.

## Quick start

```bash
# Static checks only:
./test.sh static

# Host regression (CPU only): build + unit suite + smoke + LLM smoke.
./test.sh local

# Full on-device matrix on the attached arm64 device:
./test.sh android <serial>          # e.g. ./test.sh android R5CY71BJJ9D
```

### Remote SSH device preflight

For a benchmark reached through SSH or a jump host, validate every hop before
starting a build or a timed run:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 user@gateway true
ssh -o BatchMode=yes -o ConnectTimeout=10 -J user@gateway user@target true
```

Treat `Permission denied (publickey)` on the first hop as an authentication
blocker, not as a successful reachability check. Ask for the correct key,
username, port, or authorized-key registration; never persist a supplied
password in the repository, result metadata, or generated command files.
If a non-interactive runner cannot answer the system SSH client's password
prompt, use an in-memory SSH library or credential provider and verify the
server host key; do not generate an askpass or wrapper script containing the
password.
When an older client and a newer server fail while negotiating a newly
advertised post-quantum KEX, constrain the client to a mutually supported
algorithm such as `curve25519-sha256`, but continue to pin and verify the host
key in a dedicated known-hosts file. Never turn a KEX compatibility problem
into `StrictHostKeyChecking=no`.

For a custom compiler prefix, verify that it contains an executable compiler
(for example, `bin/g++`) before the expensive run; an extracted compiler source
tree is not a usable toolchain. Record the resolved compiler path and version,
and make an ISA-specific benchmark assert that its expected optimized variant
is present instead of accepting a silently compiled scalar fallback.
Keep shared benchmark drivers compiler-agnostic: pass a site-specific compiler
through `CXX` in the invocation or runbook instead of hard-coding a local
toolchain prefix and library layout into the driver.
For normalized or reduction-based operators, check structural invariants (for
example, a Softmax output sum) in addition to elementwise absolute error. A
tiny per-element error can accumulate into a material normalization failure at
large shapes; fix the reduction precision or organization instead of relaxing
the invariant threshold.

Keep architecture-specific evidence in three non-interchangeable tiers:
host semantic emulation, target-object static inspection, and target-hardware
execution. Host emulation can expose VLEN-dependent control-flow and boundary
bugs but cannot be reported as hardware correctness. Objdump proves the emitted
static form but not runtime counts or speed; classify expected mnemonics as
required or diagnostic because compilers may fold explicit narrowing and other
source intrinsics. Preserve failed records, compiler/flags, source revision,
source and object hashes, and the exact execution tier in every summary.
An exported source tree may intentionally omit `.git`. Treat Git provenance as
optional evidence rather than a build prerequisite: record it as unavailable,
continue to hash every scoped source and harness file, and never invent a
commit identity for the copy.

Before aggregating cross-target results, establish a comparable cohort by
checking the case-key set, compiler version and flags, and source/object hashes.
Compare the target sources with an expected hash manifest immediately before
launch, not only after collection. If any scoped production source changes
during an in-flight run, preserve that result as superseded and rerun into a
fresh result directory; do not merge the stale run into the intended cohort.
If suite versions differ, report the exact common-case intersection separately
and list the missing cases instead of merging headline pass counts. Identical
per-kernel object hashes can prove that the same compiled kernel was inspected
even when an older result lacks source hashes, but they do not prove that the
harness, full binary, VLEN, governor, or other run conditions were identical;
missing provenance must remain an explicit evidence gap.

Do not trust a user-supplied platform label as hardware identity in a
multi-target experiment. Measure a distinguishing invariant such as RVV VLEN
with a target intrinsic probe, compare it with an explicit expected value
before the correctness or performance stage, and persist the expected value,
measurement, and check status in metadata. A mismatch must fail the target run
instead of producing evidence under the wrong platform name.
On a heterogeneous CPU, VLEN alone may not identify the intended cluster
because multiple microarchitectures can implement the same vector length.
Map logical CPUs to core models, pin both the probe and measured workload to an
explicit CPU set, and record the requested and effective affinity. Do not rely
on the login shell's inherited affinity or on a whole-system `/proc/cpuinfo`
model list as proof that the workload ran on the named cluster.

Do not assume a copied native compiler prefix is portable merely because
`bin/g++ --version` works. Before a long run, compile and execute both an ISA
intrinsic probe and a small C++ program that includes a threading-using standard
header such as `<memory>`. `LIBRARY_PATH` fixes link-time library lookup and
`LD_LIBRARY_PATH` fixes runtime lookup, but neither fixes libstdc++ headers
configured against an incompatible libc/pthread ABI. Prefer a compiler built on
the target or a toolchain with an explicit self-contained sysroot.
The same GCC semantic version is not a compiler-build identity. When
byte-reproducible objects matter, also retain `-dumpmachine`, verbose
configuration, compiler executable hash, assembler/binutils version, and
resolved runtime libraries. Never attribute differing object bytes to target
microarchitecture when each target compiled independently; compare required
instruction forms/counts and hardware results, and report the binary identity
gap explicitly.

Objdump parsers must be tested with the actual target compiler's output. GCC
may emit `.L*` local labels inside one function; these are not function
boundaries. For one-kernel-per-object experiments, object-scope counting is
more portable than resetting at every symbol. Mark semantic forms as required,
fail the stage when any required count is zero, and keep compiler-dependent
forms diagnostic. A strict correctness runner should still write JSONL, the
summary, provenance, and disassembly before returning non-zero; do not let
`set -e` discard the failure evidence.

For runtime policy ablations, setting an override is not evidence that the
requested implementation executed. Require the target binary to print an
observable plan or variant identity, validate it in every raw run, and fail
closed on a missing or mismatched identity. If a benchmark's JSON contains only
statistics aggregated over in-process repetitions, obtain per-process evidence
by running one retained sample per fresh process and preserving each JSON,
stdout/stderr, command, order, and exit code before computing cross-process
statistics.
When one logical plan is controlled by multiple environment variables, treat
the assignments as one atomic adapter: explicitly unset the entire override
namespace before applying the requested combination, record both the cleared
names and final assignments, and test the adapter against a deliberately stale
parent environment. Otherwise an inherited variable can silently turn a named
ablation into a different implementation.
When an experiment ships source instrumentation as a standalone unified diff,
validate the diff syntax locally (for example, `git apply --stat`) and require
`git apply --check` against the exact target tree before mutation. Put
selection tracing after the final cached decision but outside the timed hot
path, and emit it once per process; this makes the trace auditable without
turning logging overhead into part of the measured kernel.
Before extracting a benchmark result archive, compare its digest with the
producer's recorded digest, list every member, and reject absolute paths, drive
prefixes, and `..` path components. Test the traversal predicate itself with
known-safe and known-unsafe names before trusting its count, then extract only
into a fresh, dedicated directory. Preserve producer checksum manifests
verbatim: if local extraction changes the path prefix, map each recorded path
to the extracted artifact for verification instead of rewriting provenance.
For a final evidence audit, require one-to-one counts of raw logs, machine-
readable results, metadata, and exit codes; parse every result and verify one
observable plan identity per process.
Build audit assertions from the producer's actual JSON keys and CSV headers.
Inspect or schema-check a representative record before counting passes,
case-key uniqueness, or required-instruction fields; generic assumptions such
as `pass` versus `passed` can falsely report a valid result as failed.
When reanalysis runs on a host with a different Python or JSON serializer,
compare parsed fields and numeric values rather than requiring byte-identical
summary JSON. Preserve the producer summary verbatim and separately record
whether the local reanalysis is structurally and numerically equal.

Performance reports must not print a rounded small p-value as `p=0`; use
scientific notation or an upper bound. Also define cold-start precisely:
fresh-process runs are process-cold, but are not storage- or page-cache-cold
unless the harness explicitly resets and records those cache states.

When reproducing a published installation-time or calibration-overhead table,
recover the original harness's timer boundaries before introducing a cleaner
service abstraction. A legacy metric that ends after a short benchmark process
completes is not a service-ready measurement even if the paper calls it
``Restore``; retain and label the legacy-compatible metric for cross-target
comparison, and report an explicit ready-marker metric separately. Validate
`Total = Explore + Restore` for every complete sequence before aggregating,
and do not resume into an existing result cohort until the binary, model,
harness, plan adapter, affinity, and protocol hashes or fields match.

`<serial>` comes from `adb devices` (the script prefers `adbk` and falls back
to `adb`). If the device shows as `unauthorized`, the user must tap **Allow USB
debugging** on the phone first.

## Running a subset (filters)

Android mode takes an optional filter as the third argument:

```bash
./test.sh android <serial> cpu        # CPU unit + lowmem + llm
./test.sh android <serial> opencl     # OpenCL unit (image+buffer) + opencl smoke
./test.sh android <serial> vulkan     # Vulkan unit + vulkan smoke
./test.sh android <serial> gpu        # opencl + vulkan
./test.sh android <serial> unit       # all unit/op stages only
./test.sh android <serial> lowmem     # only the low-memory matrix
./test.sh android <serial> android-ci # bench + smoke + llm only (no unit/lowmem)
```

Valid filters: `all` (default) · `cpu` · `opencl` · `opencl-image` ·
`opencl-buffer` · `vulkan` · `gpu` · `unit` · `lowmem` · `android-ci`.

## Reading the result (agent-friendly)

* Each stage prints a delimited `═══ stage: <name> ═══` block, then a
  `PASS` / `FAIL` / `SKIP` line.
* A final **summary** prints `total / passed / failed / skipped` and one line
  per stage. `SKIP` is not a failure — it means the prerequisite was absent
  (e.g. a GPU library, a model, or a missing build artefact).
* **Exit code is non-zero iff any stage failed.** Gate automation on the exit
  code, not on log scraping.
* Combined stdout/stderr for every stage is saved under
  `logs/test-<UTC-timestamp>/<stage>.log` — read the named log of a failing
  stage for the trailing output. `rc=137` ≈ OOM-kill, `rc=139` ≈ SIGSEGV.

## Dynamic-shape device smoke tests

A zero exit code only proves that a backend context ran; it does not prove that
the requested input shape selected the intended dynamic context. For a
shape-sensitive device test, record and validate the runtime-observed input
shape (for example, from a backend dump manifest or the runner's input tensor)
before treating the test as dynamic-shape coverage.

## Environment variables

| Var | Mode | Meaning |
|-----|------|---------|
| `ANDROID_NDK` | android | NDK root. Falls back to `$HOME/android-ndk-r21`. |
| `ANDROID_EXTRA_CMAKE` | android | Extra cmake flags appended to the build (e.g. `-DMNN_SME2=OFF`) — handy for bisecting a backend regression. |
| `LLM_MODEL_DIR` | both | Path to an existing on-disk MNN-format LLM model. When set, that directory is used **as-is and nothing is downloaded**. Defaults to `models/<repo-basename>/`. |
| `LLM_MODEL_REPO` | both | Model repo id for the LLM smoke test. Default `taobao-mnn/Qwen2.5-0.5B-Instruct-MNN`. |
| `LLM_MODEL_SOURCE` | both | Download source when `LLM_MODEL_DIR` is unset: `huggingface` (default) or `modelscope`. |
| `LLM_MODEL_URL_BASE` | both | Override the resolve URL prefix outright (wins over `LLM_MODEL_SOURCE`). |
| `MNN_TEST_SKIP` | both | Comma list of exact test names to skip (also set per-stage via the JSON `skip` field). |

### Offline / no-network and mainland-China notes

LLM model provisioning is **lazy**: the download (or `LLM_MODEL_DIR` check) is
deferred until the `llm` stage actually runs, and a provisioning failure skips
**only** that stage. So the unit / smoke / bench stages run fine with no
network.

```bash
# Already have the model on disk → no download attempt at all:
LLM_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct-MNN ./test.sh local

# huggingface.co unreachable (e.g. mainland China) → fetch from ModelScope:
LLM_MODEL_SOURCE=modelscope ./test.sh android <serial>
```

For the built-in default model the ModelScope org is remapped automatically
(`taobao-mnn/*` → `MNN/*`); an explicitly-set `LLM_MODEL_REPO` is used verbatim.

### LLM backend/layout smoke

For backend or tensor-layout optimizations, do not stop at operator tests. Run
an end-to-end `llm_demo` correctness smoke with a short prompt and another
prompt long enough to cross backend prefill branch thresholds. This catches
real exported-graph layout bugs where an op test covers only the output format,
but the graph also changes an input tensor format.

### macOS Metal answer A/B checks

For answer-level comparisons between two macOS Metal runtimes:

* A Metal config does not prove that Metal ran. In a restricted execution
  context, `MTLCreateSystemDefaultDevice()` may return `nil`, after which
  `llm_demo` prints `Init Metal Error` / `Can't Find type=1 backend, use 0
  instead` and silently produces a plausible CPU answer. Run in a GPU-visible
  context and fail the case on either fallback marker.
* Use the same exported model, prompt bytes, backend settings and greedy
  sampler on both sides. Isolate each runtime's `tmp_path` / working directory
  so cache reuse cannot cross the A/B boundary.
* Decide the multi-line prompt contract before running. `llm_demo` normally
  treats every non-empty line as a separate prompt; if one file is intended to
  be one long prompt, flatten or use an explicitly one-line-aware runner and
  record the transformed-input checksum.
* Fixed token limits can stop in the middle of a UTF-8 token and can leave code
  answers incomplete. Exact matches are strong evidence, but low character
  similarity after an early greedy fork needs semantic review. For code
  prompts, use a large enough budget and compile/test the extracted code before
  claiming functional correctness.

## Configuring stages

Editing [`test_stages.json`](../../test_stages.json) is the supported way to
add, drop, or retune unit / lowmem / smoke / bench stages. Every parameter
(forward type, precision, gpuMode, thread count, tag, memory mode,
dynamic-quant option, KleidiAI flag, per-stage skip list, smoke model list,
benchmark args) lives there, and the `_documentation` block at the top of the
file explains every field and every `skip` entry's rationale.

* **Add a stage that runs an existing test in a new config** → add an object to
  `android.stages` (or `local.stages`). See `docs/testing.md` § "增加专门阶段".
* **Skip a known-broken test on one stage** → add its exact name to that
  stage's `skip` array **and** document why under `_documentation.skip_rationale`.
* **Add a smoke model / bench entry** → see `docs/testing.md` § "新增 smoke 模型或 bench 阶段".

## Auditing stale CI/test scripts

When asked to clean up old CI or test scripts, build a usage map before
recommending deletion:

* Prefer `git ls-files` plus targeted `rg`/`git grep` over broad filesystem
  scans, so generated build directories and local experiments do not look like
  maintained CI surface.
* Classify scripts by role: active CI entrypoints, declarative test driver,
  release/package scripts, manual benchmark helpers, third-party vendored
  tests, and local device/debug helpers.
* Treat lack of in-repo references as a "review/deprecate" signal, not proof
  of dead code; internal CI systems can invoke tracked files by convention.
  Prefer a staged deprecation plan unless a script is both unreferenced and
  clearly superseded by `test.sh` / `test_stages.json`.
* When renaming or consolidating test entrypoints, grep for both executable
  names and generated-artifact prefixes. Update CI config, `.gitignore`,
  `test_stages.json` self-documentation, developer docs, skill docs, and code
  comments in the same change so the old entrypoint disappears completely.

## Adding a new operator test

1. Write the C++ test under `test/op/` (one file, registered with
   `MNNTestSuiteRegister`). The full template + conventions are in
   [`docs/testing.md`](../../docs/testing.md) § "新增算子测试".
2. If its name prefix matches an existing stage (e.g. `op/*`), it is picked up
   automatically — no JSON change needed. Otherwise add a dedicated stage.

For deeper work on operators themselves, see the
[`add-new-op`](../add-new-op/SKILL.md) skill.

## Read next

`docs/testing.md` is the authoritative deep reference — read it for the per-stage
breakdown, the stage-object field table, and worked examples.

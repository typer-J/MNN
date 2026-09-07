# MNN 原标量 / RVV 四函数对比：SG2044 上传与测试指南

本目录测试 **PackC4、UnpackC4、ScaleAndAddBias、ReluWithSlopeChannel（逐通道 PReLU）**。
当前候选修复了四个已有 RVV 函数的符号与函数表接入，实际生产实现与 MNN 原标量函数同进程对照。
源码/脚本基线为 `e2dd1c98`，候选分支为 `codex/riscv-sg2044-first-pass`。
**目前尚未取得 SG2044 实测结果**；下面是待执行步骤，不代表全算子优化或模型加速已完成。

## 1. 需要复制哪些文件

| 测试目的 | 需要上传的内容 | 启动方式 |
|---|---|---|
| 先测本次四函数的正确性与性能，推荐先做 | 下方最小源码集合，保留目录结构 | `SKIP_INTEGRATION=1` 运行总入口 |
| 编译完整 MNN，测试相关 op 与实际派发 | 同一候选分支的完整源码 checkout | 默认运行总入口 |
| 服务器已有同版本完整候选 checkout | 核对版本后直接用其中的脚本 | 不必重复复制依赖 |

**只上传 `run.py` 或本目录是不够的。** 测试会从 MNN 源码目录读取原标量函数、生产 RVV 函数和依赖头文件。
服务器旧 MNN 中的同名文件不能自动视为本次候选，建议使用独立目录保存本次测试源码。

### 1.1 最小源码集合

所有路径均相对于本机本次候选的 MNN 根目录。服务器新目录内必须保持相同相对路径，不能把文件拍平：

```text
mnn-rvv-c4/
├── LICENSE.txt                                  # 保留源码许可证
├── benchmark/rvv_scalar_compare/
│   ├── run.py                                   # 必需：提取、编译、运行、汇总
│   ├── benchmark.cpp                            # 必需：正确性与交错计时
│   ├── probe.cpp                                # 必需：RVV/VLEN/hwprobe 探针
│   ├── sg2044_validate.sh                        # 推荐：服务器总入口
│   ├── integration.sh                           # 附带：仅完整源码可使用
│   ├── README.md                                # 本文
│   └── .gitignore                               # 可选：忽略结果与缓存
├── source/
│   ├── backend/cpu/
│   │   ├── CPURuntime.cpp                       # 必需：运行时源码留档
│   │   ├── compute/CommonOptFunction.cpp         # 必需：原标量函数来源
│   │   └── riscv/rvv/
│   │       ├── MNNRvvC4Functions.hpp             # 必需：四函数声明
│   │       ├── MNNPackC4.cpp                     # 必需：生产 RVV 实现
│   │       ├── MNNUnpackC4.cpp                   # 必需
│   │       ├── MNNScaleAndAddBias.cpp            # 必需
│   │       └── MNNReluWithSlopeChannel.cpp       # 必需
│   ├── math/Vec.hpp                             # 必需：原 Scale 使用的实际 Vec
│   └── core/
│       ├── Macro.h                              # 必需
│       └── SimdHeader.h                         # 必需
└── include/MNN/MNNDefine.h                       # 必需
```

其中 runner 实际读取 **14 个必需文件**；加总入口、README、许可证即为 17 个推荐文件，
再附带 `integration.sh` 与 `.gitignore`，上面的完整清单为 **19 个文件**。
`CPURuntime.cpp` 虽然不参与这个小程序的编译，仍会被快照读取，不能漏传。
`CommonOptFunction.cpp` 只被提取其中四个原函数和两个 pack 模板，函数体不改写，不会整文件编译。

无需上传本机的 `.git/`、`.codex_tmp/`、`results/`、缓存、Windows 可执行文件、`.o` 或已有 `libMNN.so`。
这些测试会在服务器用 GCC 14 重新编译。没有 `.git` 的包也能运行，源码 SHA256 仍记录，Git 身份记为不可用。

### 1.2 完整 MNN 集成需要什么

最小包没有完整 `CMakeLists.txt`、测试、Schema 和其他构建依赖，**不能执行完整 MNN 集成**。
小包即使附带了 `integration.sh`，也必须设置 `SKIP_INTEGRATION=1`。

完整集成应使用同一候选分支的完整源码，避免将少量新文件直接覆盖到旧仓库后就当成完整版本同步。
服务器可访问仓库时，在一个不存在的新目录克隆：

```bash
git clone --depth 1 --branch codex/riscv-sg2044-first-pass \
  https://github.com/typer-J/MNN.git /path/to/new/MNN-rvv-validation
git -C /path/to/new/MNN-rvv-validation rev-parse HEAD
```

记录实际提交号。离线复制完整源码也应从该候选版本准备到新目录，保留原仓库的在制修改。
本脚本的集成范围是相关 op，不是全部 MNN 测试或模型回归。

## 2. MobaXterm 上传步骤

1. 在已登录目标服务器的会话中，通过左侧 SFTP 文件面板上传源码 ZIP 和对应 `.sha256` 文件。
2. 在上传目录执行 `sha256sum -c <源码包名>.sha256`，确认 `OK`。
3. 将源码解压到原 MNN 仓库之外的新目录。可用已有 Python 3，无需安装 unzip：

   ```bash
   python3 -m zipfile -e /path/to/upload/mnn-rvv-c4-upload.zip /path/to/new/extraction-directory
   ```

4. 进入解压后的 `mnn-rvv-c4/`；若包中附带 `SHA256SUMS`，执行 `sha256sum -c SHA256SUMS`。
5. 核对 `benchmark/rvv_scalar_compare/run.py` 与 `source/backend/cpu/compute/CommonOptFunction.cpp` 都存在。

也可以按 1.1 的清单逐文件上传，保留路径与 LF 换行。优先传压缩包，避免传输时改变 shell 脚本换行。
源码 hash 不匹配时重新上传，不要修改校验清单来掩盖错误。

## 3. 设置路径、检查 GCC 14

下面三项是会话参数，将占位路径替换为实际值。`ORIGINAL_REPO` 是服务器原有 MNN，
`CANDIDATE` 是刚解压或克隆的新目录；两者不能相同，候选也不能放在原仓库之下。
本次源码包附带的服务器专用说明可以直接填入会话路径；公共 README 使用占位符。

```bash
unset CC CXX
export ORIGINAL_REPO='/path/to/existing/MNN'
export GCC14_PREFIX='/path/to/gcc14'
export CANDIDATE='/path/to/new/mnn-rvv-c4'
cd "$CANDIDATE"

uname -m
python3 --version
test -x "$GCC14_PREFIX/bin/gcc"
test -x "$GCC14_PREFIX/bin/g++"
"$GCC14_PREFIX/bin/g++" --version
"$GCC14_PREFIX/bin/g++" -dumpmachine
```

期望架构为 `riscv64`、编译器主版本为 14、目标 triple 以 `riscv64` 开头。
只有 GCC 源码、没有可执行 `bin/g++` 的目录不能直接使用。
总入口还会编译运行含 `<memory>` / `<thread>` 的程序，检查标准库和线程 ABI，再执行真实 RVV 探针。

小包需要现有 Bash、Python 3 标准库、RISC-V GCC 14 的 C/C++ 编译器及运行库；无需 pip 包、CMake 或 GDB。
完整集成另需 CMake、构建工具（通常 make）、ldd 和 taskset。GDB 可选，脚本不会安装依赖。

若编译器布局不是 `bin/gcc` 和 `bin/g++`，可显式设置单个可执行文件路径：

```bash
export CC='/actual/path/to/riscv64-linux-gnu-gcc'
export CXX='/actual/path/to/riscv64-linux-gnu-g++'
```

总入口的 CC/CXX 不接受附带 flags 的命令字符串。已设置的 `CXX` 优先于 `GCC14_PREFIX`。
标准 prefix 配置块已清理旧变量；如果随后按非标准布局显式设置了 CC/CXX，后续运行时保留这两个变量。

## 4. 推荐：先跑最小包的函数级测试

设置完上一节的环境变量，在候选根目录执行：

```bash
SKIP_INTEGRATION=1 bash benchmark/rvv_scalar_compare/sg2044_validate.sh
run_rc=$?
printf 'validation_exit=%s\n' "$run_rc"
```

执行顺序为：目录保护与环境检查 → GCC 14/标准库检查 → RVV/VLEN/hwprobe 探针 →
四函数正确性（不计时）→ **3 个新进程的性能对比**（每个 case 默认交错 7 轮）→ 跳过完整集成并打包。
前面阶段失败会停止后续阶段，保留错误和已有结果。

默认结果放在 `benchmark/rvv_scalar_compare/results/session-<UTC时间>/`，不需要预先创建目录。
也可指定一个候选目录内、尚不存在的路径：

```bash
SKIP_INTEGRATION=1 bash benchmark/rvv_scalar_compare/sg2044_validate.sh \
  "$CANDIDATE/benchmark/rvv_scalar_compare/results/run-001"
```

复跑改成 `run-002`，或省略路径。已有输出目录或同名压缩包不会被覆盖。
最前面的目录保护失败时不创建结果；上传、Git 克隆等脚本之外的失败也不产生测试结果包。

### 4.1 绑核与参数

默认选择当前进程允许使用的第一个逻辑 CPU，并约束探针和性能进程。先查看可用集合：

```bash
python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))'
```

例如确认 CPU 8 在集合中后，可以指定：

```bash
CPUSET=8 PERF_PROCESSES=3 BENCH_ROUNDS=7 BENCH_SAMPLE_MS=1 \
  SKIP_INTEGRATION=1 bash benchmark/rvv_scalar_compare/sg2044_validate.sh
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `EXPECT_VLEN` | `512` | VLEN 位数的预期值，必须由实际探针确认 |
| `PERF_PROCESSES` | `3` | 新性能进程数 |
| `BENCH_ROUNDS` | `7` | 每个 case 的交错测量轮数，至少 2 |
| `BENCH_SAMPLE_MS` | `1` | 较快一侧单批次的目标校准时长，有限正数 |
| `CPUSET` | 第一个允许的 CPU | 逗号分隔的逻辑 CPU，例如 `8` 或 `8,9,10,11`，不用 `8-11` |
| `JOBS` | `4` | 完整 MNN 的并发编译任务数，不是微基准线程数 |
| `SKIP_INTEGRATION` | `0` | 最小包必须显式设置 `1` |
| `PYTHON` | `python3` | 已安装的 Python 解释器 |

单核测速优先使用一个 CPU；多个 CPU 允许进程在集合内迁移。绑核不独占核心，脚本也不关闭其他任务或修改调频。
负载、频率、温度记录是快照，不是持续监控或受控环境保证。高负载或波动较大时应保留本次数据，在稳定条件下复跑。
默认 512 位是待验证预期；实测不匹配时先核对目标机器和 ISA，不能直接修改预期来掩盖不匹配。

## 5. 完整源码：运行 MNN 集成回归

确认候选目录是完整 checkout，再运行：

```bash
cd "$CANDIDATE"
unset SKIP_INTEGRATION
JOBS=4 bash benchmark/rvv_scalar_compare/sg2044_validate.sh
```

函数级阶段成功后，分别创建 `integration/build-scalar/`、`integration/build-rvv/`，编译两套完整产物。
开启测试、LLM、LOW_MEMORY、TRANSFORMER_FUSE 和线程池，关闭 IME2、fast-math 与 OpenMP。
两套构建的通用 TU 使用 `rv64gc`、禁自动向量化和 FMA contraction；RVV 构建仅给 RVV TU 追加 `rv64gcv`。

每套测试 `op/convert`、`op/scale`、`op/prelu`、`op/relu`，各用 1/4 线程。
实际参数为 `run_test.out <前缀> 0 1 <线程数> sg2044_integration 0`，即 CPU、High 精度、Normal 内存模式。
测试按前缀匹配，Scale/PReLU 也包含对应 Int8 用例。脚本检查真实退出码、准确测试名、非空 passed 与零失败。

服务器已有 GDB 时，另跑不计时的 Scale/PReLU，记录四个 `_RVV` 函数命中次数及调用栈。
`op/convert` 可能走 Transpose 路径，不能独立证明本次 Pack/Unpack 派发生效。
派发结果区分 `verified`、`not_verified`、`blocked`、`failed`。

这些小 op 的 4 线程通过不代表多核性能提高；若 CPUSET 只有一个 CPU，四个 worker 仍只在该 CPU 上调度。
模型没有被选择或下载，`model_stage=not_run`；模型正确性、吞吐、LLM 短/长 prompt 仍需另测。

已有函数级结果、只补集成时也可单独执行：

```bash
CC="$GCC14_PREFIX/bin/gcc" CXX="$GCC14_PREFIX/bin/g++" JOBS=4 \
  bash benchmark/rvv_scalar_compare/integration.sh \
  "$CANDIDATE/benchmark/rvv_scalar_compare/results/integration-only-001"
```

该入口仍需要已设置的 `ORIGINAL_REPO`，输出目录必须新建于候选内；不重跑 VLEN/函数级探针，也不自动打包。

## 6. 结果文件与加速比怎么看

总入口最后打印 `Archive:`、`SHA256:`、`Results:` 和 `Status:`，正常结果结构如下：

```text
results/session-<时间>/
├── summary.json                    # 总阶段状态、错误与集成覆盖限制
├── exitcode                        # 总入口退出码
├── preflight.context.json          # CPU、负载、频率/温度快照
├── rvv-probe-run.log                # 实测VLEN、RVV运算与hwprobe
├── correctness/                    # 不计时，没有性能CSV属正常
│   ├── metadata.json
│   └── results.jsonl
├── performance-1/                  # performance-2/、performance-3/同结构
│   ├── metadata.json               # 版本/hash、编译参数、CPU及退出码
│   ├── results.jsonl               # 正确性与逐次原始计时
│   ├── samples.csv                 # 每次采样
│   ├── comparison.csv              # 每个case的对比
│   ├── source_snapshot/
│   ├── build.log
│   └── run.log
└── integration/                    # 仅完整集成时存在
    ├── summary.json
    ├── logs/                       # 每一步命令、日志、退出码
    ├── evidence/                   # 测试解析、库/二进制hash与构建检查
    ├── build-scalar/
    └── build-rvv/
```

失败时结构可能不完整，先看总 `summary.json` 的 `error`、`failure.log` 与对应阶段 `.log`。
退出码 0/总状态 passed 只表示所执行检查通过；小包会明确记录 `integration=explicitly_skipped`。
它不表示全部优化或性能验收完成。

函数级 `metadata.json` 应有 `status=passed`、`summary.cases > 0`、`summary.failed=0`；
性能阶段应有非零 samples，正确性-only 则是零 samples。

| `comparison.csv` 列 | 含义 |
|---|---|
| `op`、`area`、`channels` | 函数和输入规模 |
| `src_area_stride`、`dst_area_stride`、`inplace` | 布局和原位条件 |
| `scalar_median_ns`、`rvv_median_ns` | 每次调用中位耗时，纳秒 |
| `scalar_mad_ns`、`rvv_mad_ns` | 绝对中位差，观察波动 |
| `scalar_min_ns` / `scalar_max_ns`、`rvv_min_ns` / `rvv_max_ns` | 本进程各轮最小/最大耗时 |
| `speedup_scalar_over_rvv` | 标量中位耗时 / RVV 中位耗时 |

加速比大于 1 表示该 case 的 RVV 更快，小于 1 表示更慢；2 表示耗时减半。零工作量不报告加速比。
按同一 shape/stride/原位条件检查三个进程是否复现，结合 MAD 和负载判断，不挑最快一轮或把不同 case 平均成全库提升。

### 6.1 下载哪些结果

通过 MobaXterm SFTP 下载末尾打印的 **结果 `.tar.gz` 与同名 `.sha256`**，将两个文件交回分析。
在包所在服务器目录可先执行 `sha256sum -c <结果包名>.sha256`。
无需复制编译器或整个服务器 MNN；失败记录同样需要保留，不能只返回成功截图。

结果包位于结果目录的同级，包含日志、源码快照、微基准产物和 JSONL/CSV，排除两个体积较大的 MNN build 目录，
但保留其编译记录、二进制/库 hash 和动态库解析日志。结果内含本次路径与环境信息，按实验数据处理。
归档失败时保留整个结果目录和错误输出；中断、SIGKILL 或断电可能来不及生成最终摘要。

## 7. 单阶段排查与常见问题

上传后可先做不需要编译器的完整性检查，这会验证 14 个文件是否可读以及原函数能否正确提取：

```bash
python3 benchmark/rvv_scalar_compare/run.py --extract-only
```

只运行目标机正确性：

```bash
CXX="$GCC14_PREFIX/bin/g++" python3 benchmark/rvv_scalar_compare/run.py --correctness-only
```

直接 `run.py` 不负责原仓库隔离、负载/温度快照、多进程编排和结果打包，推荐默认使用总入口。
`--out` 可指定新目录，省略则自动产生时间戳目录；`--cpu` 是 `--cpus` 的别名。

| 现象 | 处理 |
|---|---|
| 缺少 cpp/头文件 | 按 1.1 核对相对路径；只传 benchmark 目录不够 |
| `Expected exactly one definition` | 检查是否混入旧版、文件截断或误传，重新使用同一候选源码 |
| `ORIGINAL_REPO is required` / 隔离失败 | 设置真实原仓库路径，把候选放到它之外，不要伪装变量来绕过检查 |
| 输出目录已存在 | 换新编号或使用默认时间戳，保留旧结果 |
| `bin/g++` 不存在/不是14 | 确认安装而非源码目录；核对显式 CC/CXX 和遗留环境变量 |
| `<memory>` / `<thread>` 失败 | 看 memory-build/run 日志，检查标准库、libc/pthread ABI 和运行库路径 |
| `libstdc++.so` / `GLIBCXX_*` 错误 | 按该编译器真实安装位置配置运行库路径；路径设置不能修复 ABI 不兼容，不替换系统库 |
| `riscv_vector.h` 缺失 | 确认完整的 RISC-V GCC 14 安装和实际编译器路径 |
| `Illegal instruction` / VLEN不匹配 | 停止测速，核对目标机器、ISA、内核和预期值，保留探针日志 |
| RVV成功但hwprobe不报告V | 直接调用可执行不等于 MNN 会启用 RVV，核对 runtime 能力门和派发 |
| 小包最后缺CMakeLists或集成失败 | 设置 `SKIP_INTEGRATION=1`；完整集成另准备完整源码 |
| GDB不存在/ptrace被限制 | op结果可保留，派发记为未验证/受阻，不能声称四个入口已命中 |
| `$'\r'` / bad interpreter / heredoc错误 | 换行被改变，重新上传未改动ZIP并核对源码hash |
| 没有comparison.csv | correctness-only属正常；性能运行则查看是否提前失败 |
| RVV比原标量慢 | 保留全部case数据，核对三进程、MAD与负载，再按实测热点调整实现或回退 |

## Technical reference: baseline and measurement details

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

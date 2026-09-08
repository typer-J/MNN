# MNN 面向 SG2044 的 RISC-V 适配与优化顺序

更新日期：2026-09-08。目标硬件：SG2044；目标编译器：GCC 14。

本路线图先审计算子库的实际接入情况，再安排 pack、执行器、Attention/KV、线程和内存优化。当前结论是：**仓库已有 RVV 实现，但有效覆盖并不完整，部分实现存在符号或接口未接通的问题。第一步应修正确性与接入，并优化热点算子；第二步集中优化 pack 和数据搬运。**

## 1. 审计基线与交付状态

| 项目 | 本次记录 |
|---|---|
| 同步前本地 master | `54706985` |
| fetch 后 upstream/master | `bef71b9756a2c77549eddbe33eb97290e3b16602`；本次远程跟踪引用前进 4 个提交 |
| 同步前与最新 upstream 的关系 | 本地独有 3 个提交、upstream 独有 185 个提交，采用合并保留双方历史 |
| 合并并同步 origin/master 后的审计基线 | `e44af4a6` |
| 首批生产代码候选 | `fe7f86c6`，工作分支 `codex/riscv-sg2044-first-pass`；四函数目标微基准原始包已核验，完整 MNN 验收待完成 |
| 本文源码结论 | 针对以上合并基线的静态审计，固定版本链接用于避免后续修改混淆现状 |
| 第一批候选工作 | C4 pack/unpack、Scale、PReLU 的 RVV 接线修复已实现；静态编译/签名检查通过，目标四函数矩阵与三个性能进程均回传成功摘要；标量对照见 `benchmark/rvv_scalar_compare/` |
| SG2044 正确性与性能结果 | 四函数1320配置正确性及采样已通过；原始包核验发现函数/尺寸相关收益与退化，尚无完整 MNN 派发/模型结论 |

“已有实现文件”“链接进产物”“实际命中”“正确性通过”“在目标板更快”是五种不同状态。
基线 RVV 顶层有 79 个 `.cpp`，IME2 子目录有 6 个 `.cpp`；这些数量不代表算子优化覆盖率。
后续记录必须另列候选提交和验证状态，不能把候选修复追记为 `e44af4a6` 已具备的能力。

## 2. 硬件范围与前置基线

本轮默认 **Linux riscv64、标准 RVV 1.0、`rv64gcv/lp64d`、IME2 OFF、fast-math OFF**。
SG2044 不走 SpacemiT IME2 路线。实际 VLEN、可见 CPU、内核能力和频率策略由板端探针确认，不从芯片名称推断。
FP16/BF16 必须独立验证：探测到硬件扩展不等于 MNN 已具备完整的对应精度执行路径。

开始计算核优化前，保存以下不含连接信息的实验坐标：

- SG2044 的系统架构、内核、运行时 VLEN/VLENB、hwprobe 结果、在线 CPU 与实际绑核集合。
- GCC 14 的完整版本、sysroot、实际 `-march/-mabi`、构建选项和编译命令。
- 候选与基线 commit、二进制及动态库摘要，确保加载本次构建产物。
- 模型及权重量化格式，prefill/decode shape、线程数、batch、上下文长度和采样方式。
- 温度、频率、系统负载、冷启动/稳态区分；真实板端数据单独保存，本文只维护方法与状态。

构建和运行时已有的基础不要重复“从零实现”：

| 基础 | 当前事实 | 后续验证 |
|---|---|---|
| RVV 构建门 | `MNN_USE_RVV` 默认 OFF；processor 匹配 riscv64 或 `ARCHS=riscv64` 时启用 | 检查 MNNRVV target 与实际编译命令，不能只检查 cmake 成功 |
| 硬件探测 | 已有 `riscv_hwprobe`，查询所有在线 CPU 的能力交集 | 验证内核支持、syscall 失败和受限环境回退；不要直接强制 RVV=true |
| 线程等待 | ThreadPool 已有 RISC-V pause 指令编码及 spin/yield/block 路径 | 在 SG2044 测同步成本，不能把“增加 pause”列为尚未实现项 |
| CPU 分组 | 已按 cpufreq policy 分组，有 CPU 数 fallback | 无 cpufreq 时 group 可能仍为空，需确认显式 CpuIds 和绑核真正生效 |

证据：[RISC-V CMake](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/riscv/CMakeLists.txt#L7)、[hwprobe](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/CPURuntime.cpp#L1554)、[pause](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/ThreadPool.cpp#L48)、[CPU 分组](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/CPURuntime.cpp#L1653)。

## 3. 算子和 pack 库排查结果

### 3.1 计算核和算子接入

| 范围 | 基线现状 | 优先处理的问题 |
|---|---|---|
| FP32 packed GEMM | 已通过 CoreFunctions 注册 RVV；pack mode 为 E/L/H=16/1/4 | 核查 tile、tail、真实 VLEN 利用和后处理成本 |
| FP32 E=1/GEMV | RVV Remain 的 E=1 分支是四个标量 accumulator 的 C++ 循环；独立 E_1/H_1 字段仍通用 | 看 GCC 14 最终汇编，定位 decode；保留 ABI 后评估 RVV GEMV |
| INT8 普通 kernel | 已注册 RVV，含 K16 widening multiply/reduction | 测 reduction、权重复用、后处理和小 M；不能外推宽 VLEN 收益 |
| INT8 Fast | 基线 Fast 字段没有标准 RVV 覆盖 | 实际 Execution 可能选择 Fast 绕过普通 RVV kernel，先统计调用 |
| INT4/W4 | 标准 RVV 无专用 W4 注册，保留通用 W4 内核 | 优先按真实模型需求补 W4 |
| W2/W3 | 标准 RVV 对应压缩内核指针仍为空，使用 loader 展开后的 INT8 路径 | 不可误计为已有压缩 W2/W3 GEMM；按模型需求评估保持压缩的专用内核 |
| 平面 Norm/SiLU/Exp/近似 GELU | 有通过显式契约接入的 RVV 实现 | 仍需检查小 shape、尾部、误差和编译器代码；标准 GELU 仍含标量 erf |
| C4/residual Norm | `MNNNormPacked` 仍绑定通用实现 | 不可用平面 Norm 的向量化代表整个 LayerNorm/RMSNorm 已覆盖 |
| 当前 Softmax | RVV 文件是旧 3 参数函数；当前函数表使用 11 参数接口，旧函数未接入 | 实现当前 mask、running max/sum、scale 更新语义，再注册和验收 |
| Binary/Unary/Reduction | 多数仍通过通用选择器；Vec 模板没有 RVV 特化 | 按真实热点补充，避免无证据地整体改写 Vec |
| LinearAttention | RankOneUpdate、DualMatVec、DecayRankOneUpdate、FusedGatedDelta 等仍为 Default | 若目标模型包含线性注意力，按逐节点 profile 提前处理 |

证据：[FP32 注册](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/CommonOptFunction.cpp#L5137)、[E=1 分支](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/riscv/rvv/MNNPackedMatMulRemainFP32.cpp#L29)、[INT8 注册](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/Int8FunctionsOpt.cpp#L2805)、[低比特实际选择](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/ConvInt8TiledExecutor.cpp#L931)。
当前 [Softmax 实现](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/CommonOptFunction.cpp#L2901) 与 [旧 RVV Softmax](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/riscv/rvv/MNNSoftmax.cpp#L4) 的签名不同。

### 3.2 必须先修复的符号与契约问题

基线部分 RVV 文件没有包含声明契约的头文件，也没有沿用 `extern "C"` linkage。
它们定义 C++ 修饰符号，而通用函数表引用头文件声明的 C 符号；仅看到同名函数或函数表赋值不能证明接通。

| 实现文件（相对 `source/backend/cpu/riscv/rvv/`） | 风险及处理 |
|---|---|
| `MNNPackC4.cpp`、`MNNUnpackC4.cpp`、`MNNPackC4ForMatMul_A.cpp` | 检查 C/C++ 符号、声明和最终绑定，先接通已有代码再讨论 pack 提速 |
| `MNNScaleAndAddBias.cpp`、`MNNReluWithSlopeChannel.cpp` | 相同 linkage 问题，优先形成小范围修复和原标量对照 |
| `MNNConvRunForLineDepthwise.cpp`、`MNNMatrixAdd.cpp`、`MNNMatrixSub.cpp` | 核对 `ConvOpt.h` 契约和通用定义是否仍存在，避免误计覆盖 |
| `MNNPackC2.cpp` | RVV 定义 float 版本，契约与已发现调用是 double 版本；尚未找到 float 主路径消费者 |

源证据：[C linkage 契约](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/CommonOptFunction.h#L31)、[PackC4 定义](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/riscv/rvv/MNNPackC4.cpp#L3)、[A pack 定义](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/riscv/rvv/MNNPackC4ForMatMul_A.cpp#L6)、[C2 契约](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/CommonOptFunction.h#L78)。

修复优先采用显式 `_RVV` 符号、包含契约头和按能力覆盖函数表，保留原标量定义供回退与测试。
不要批量补 `extern "C"`：通用同名定义若未排除，会产生重复定义；误扩 `#ifndef` 范围又会排除无关函数。
用 GCC 14 的 `nm/readelf/objdump`、链接结果和临时调用计数验证；本机目标 IR/符号检查只是辅助证据。

### 3.3 pack、Attention 与 KV 当前状态

| 路径 | 已有源码能力 | 待解决的问题 |
|---|---|---|
| FP32 C4 pack/unpack | 沿 area 动态 vl，使用 strided load/store，pack 处理尾通道补零 | 先修 linkage；再测 m8、segment 访存、短 shape 与长 area |
| A pack | 主通道组 RVV、固定内部块 1024，`l%4` 尾通道标量 | 先修 linkage，再测寄存器压力和尾部；与 GEMM tile 同步 |
| B pack | `_RVV` 入口已显式注册，支持转置 | `y_len<=4` 限制单次向量长度；全量 memset 后覆盖有效数据增加写流量 |
| Transpose/int8/int16 转换 | CoreFunctions 有通用路径 | 不等同于 float C4 pack 已覆盖；按热点补直接转换 |
| RVV Attention Execution | 已注册扩展，调用 QK、Softmax、PV，并可直接 C4 输出 | fast path 仅有限的非 Flash FP32 decode，QK/PV E=1 仍进入上述 C++ 分支 |
| 通用 Flash/GQA | 有 GQA 合批，可复用已注册 RVV GEMM | 旧 RVV Softmax 未接当前接口；需按 SG2044 的带宽、线程和 context 调整 |
| KV append | 对齐 K 可走 C4 pack；C4 V 有批量 memcpy；非量化 KV 可并发更新 | K pack 是否向量化取决于 linkage 修复；decode/非对齐与量化写入另查 |
| KV 扩容与 chunk | 已有物理 chunk、单线程宽 chunk、按前沿 tile 补零等设计 | 在板端测实际 RSS、复制量和长序列边界，不照搬其他架构实测参数 |

Attention 专用门禁见 [MNNRvvAttentionFunctions.cpp](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/riscv/rvv/MNNRvvAttentionFunctions.cpp#L65)：seqLen=1、非 Flash、K/V 不量化、bytes=4、pack/hP=4、lP=1、因果且无 sinks 等。
不满足时进入通用 CPUAttention；通用路径仍可使用已接入的 RVV GEMM，不能简单标为“Attention 全部标量”。
调用和布局证据：[Dense 卷积](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp#L305)、[MatMul](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/CPUMatMul.cpp#L103)、[K append](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/CPUKVCacheManager.cpp#L787)、[KV 并发](https://github.com/typer-J/MNN/blob/e44af4a6/source/backend/cpu/CPUKVCacheManager.cpp#L1019)。

## 4. 推荐执行顺序

| 顺序 | 阶段 | 主要产出 | 进入下一阶段的条件 |
|---|---|---|---|
| 前置 | SG2044/GCC 14 基线与路径探针 | 可重复构建、标量与 RVV 基线、真实 profile | 环境可复现，能证明执行路径 |
| 第一步 | 全量算子盘点、派发/符号修复、热点计算核 | 覆盖表和首批实际接通的优化 | 对应标量对照及板端正确性通过 |
| 第二步 | pack/unpack/transpose/im2col、量化与搬运融合 | 降低 pack 时间和中间流量 | producer/consumer ABI 与尾部完整通过 |
| 第三步 | Execution 算法选择、tile、融合与直接输出 | 减少调度前后的额外工作 | op 收益能在真实模型复现 |
| 第四步 | Attention/GQA/KVCache 全链路 | 改善 prefill、长 context 与 decode | 长序列、量化与线程组合通过 |
| 第五步 | 线程拓扑、调度与内存系统 | 合适的并行粒度、稳定 RSS 与带宽 | 线程 sweep 和冷/热启动无回归 |
| 第六步 | 精度、新 pack 契约及可选架构扩展 | 硬件支持且有收益的增量能力 | 完整 fallback 与跨路径门禁通过 |
| 第七步 | 回归矩阵、CI、体积与发布验收 | 可持续维护的 SG2044 支持 | 正确性、性能、体积和链接全部可复现 |

第一步的“全量”指全量盘点和状态归类，不要求重写所有冷门算子后才优化 pack。
pack 契约从第一步就必须明确；第二步集中实施数据搬运优化。若 profile 显示同步或 KV 已是瓶颈，可提前对应阶段。

### 第一步：全量算子盘点、接通已有代码，再做热点算子

1. 按模型 Op → CPU Execution → 函数表字段 → 声明/定义符号 → 最终 kernel 建立覆盖表。
2. 每项记录 dtype、shape、普通/Fast/低比特分支、pack、支持条件、回退和实际命中次数。
3. 首批接通 C4 pack/unpack、Scale、PReLU；先保持原算法与接口，使用原标量函数逐元素对照。
4. 继续清点 A pack、Depthwise、MatrixAdd/Sub 等 linkage，修复旧 Softmax 接口适配。
5. 按模型 profile 优先处理 E=1 GEMV、INT8 Fast/W4、C4 Norm、Softmax、卷积及线性注意力辅助核。
6. 其余 Binary/Unary/Reduction、Pool/Resize、图像处理按热点补齐，并记录保留通用实现的理由。

主要修改位置：`source/backend/cpu/riscv/rvv/`、`compute/CommonOptFunction.h/.cpp`、`compute/Int8FunctionsOpt.cpp`、`compute/ConvOpt.h/.cpp`；必要时修改对应 Execution 选择逻辑。
测试位置：`test/op/` 复用算子语义用例，针对符号接入与实际 packed ABI 增加最小回归；集成微基准放 `test/speed/`，首批可独立运行的原标量对照在 `benchmark/rvv_scalar_compare/`。
依赖：GCC 14 工具链和硬件路径已确认；函数表布局、签名、字节 stride、尾部和后处理语义已明确。
验收：原标量对照通过、符号唯一且实际命中、RVV OFF 构建仍通过、SG2044 对应 op 通过；性能数字必须来自板端。
可复用：`op/scale`、`op/relu`、`op/prelu`、`op/matmul`、`op/matmulBConst`、`op/convolution/conv2d`、`op/convolution/weighti8i4conv2d`、`op/layernorm/c4`、`op/layernorm/c4_binary`、`op/softmax`。

### 第二步：pack、布局转换、im2col 与量化搬运

1. 维持当前 FP32 `pack=4/bytes=4` 和 GEMM E/L/H=16/1/4 基线，分别计时 A pack、B pack、计算和输出转换。
2. 修复接入后再比较 C4 strided 与 segment 访存、不同 LMUL、大小 area；不能假定 m8 一定最快。
3. 优化 A pack 非四通道尾部、连续输入快速路径和局部分块，检查汇编 spill。
4. 优化 B pack 的多 panel 搬运、转置和 padding 清零，保留实际 kernel 读取区域的初始化保证。
5. 缓存不可变权重的 reorder，减少动态 shape 引起的重复 pack；小矩阵保留轻量入口。
6. 测 im2col → pack 的重复访存；低比特评估 absmax/动态量化/A pack/sum(A) 合并，保留量化语义。
7. 将 NHWC/NC4、int8/int16 转换和 direct output 纳入 profile；C2 先确定消费者再决定接入或清理。

主要修改位置：`rvv/MNNPackC4.cpp`、`MNNUnpackC4.cpp`、`MNNPackC4ForMatMul_A.cpp`、`MNNPackForMatMul_B.cpp`、`generalIm2col.cpp`、`compute/DenseConvolutionTiledExecutor.cpp`、`compute/ConvInt8TiledExecutor.cpp`、`CPUMatMul.cpp`。
依赖：第一步对应入口接通并通过数值测试；packer 和主/tail kernel 必须共享明确的布局契约。
验收：非连续 stride、C/H/K 尾部、转置、偏移、padding 和多线程偏移正确；比较总搬运字节、pack+compute 总耗时与 scratch 峰值。
可复用：`op/convert`、`op/transpose`、`op/matmul`、`op/matmulBConst`、卷积测试；`speed/Convert`、`speed/Transpose`、`speed/MatMulBConstTest`。

### 第三步：Execution、tile 选择、融合和直接输出

1. 分开设计大 M prefill 与 M=1 decode，按 SG2044 数据选择 E/H tile、K 分块、输出 panel 和线程粒度。
2. 对小 shape 避免 pack、临时 C 和任务分发成本主导；按测量决定何时使用 GEMV、直接卷积或轻量路径。
3. 联合调优 Dense/Depthwise/Winograd 等已有算法选择，不能仅因为 RVV kernel 存在就强制采用某算法。
4. 在 epilogue 融合 bias、scale、clamp、激活与最终布局写入，减少先写临时 C 再转换。
5. 检查 resize/Geometry 重建是否导致 Execution 和常量权重 reorder 重复创建，保留可复用资源。

主要修改位置：`CPUMatMul.cpp`、`compute/DenseConvolutionTiledExecutor.cpp`、`compute/ConvolutionFloatFactory.cpp`、`compute/ConvInt8TiledExecutor.cpp`，必要时追踪 `source/geometry/` 的相应公开实现。
依赖：核和 pack 已有可靠基线；不能把单独更换 tile 当成不影响 ABI 的常量修改。
验收：主 tile 与 tail、不同 B 生命周期、bias/clamp、线程分片都正确；op 与真实模型端到端均复测，冷启动单列。
可复用：卷积族、`op/matmulBConst`、`speed/MatMulTest`、`speed/MatMulBatchTest`、`speed/MatMulBConstTest`。

### 第四步：Attention、GQA 与 KVCache

1. 先接入当前 11 参数 Softmax，并覆盖 mask、running max/sum、updateScale 与 in-place 行为。
2. 分别验证 RVV 非 Flash decode、通用 Flash prefill/decode、量化 KV 路线；统计哪些 shape 命中专用入口。
3. 优化 QK/PV 的 E=1 热点和 scale/mask/softmax 访存；评估 GQA query heads 合批以共享 K/V 读取。
4. 优化 decode 和非对齐 KV append、量化 K/V 写入、缓存扩容复制与长上下文 scratch 分配。
5. 分别调逻辑 attention block 与物理 V chunk；改尺寸时同步地址、bExtraStride、padding 和量化 PV 调用点。
6. 扩展 fast path 覆盖前先证明每个新门禁组合正确；不满足条件继续走已验证的通用路径。

主要修改位置：`rvv/MNNRvvAttentionFunctions.cpp`、`rvv/MNNPackedMatMulRemainFP32.cpp`、`CPUAttention.cpp`、`CPUKVCacheManager.cpp/.hpp`、当前 Softmax/量化 KV 的函数表入口。
依赖：GEMM/GEMV、pack 和 Softmax 契约正确；量化 accumulator、scale/zero-point 与 cache metadata 按实际布局比较。
验收：GQA/MQA、C4/非 C4、head dim tail、Flash 开关、KV 量化和多线程；长度跨 64、2048、4096 边界；固定采样短生成与长 prompt 都通过。
可复用：`op/attention`、`op/attention_nocache_mask`、`op/attention_kvblock`、`op/attention_c4`、`op/attention_c4_tail`、`speed/attention`、`speed/attention_threads`。

### 第五步：线程、拓扑、内存带宽与启动

1. 按真实 online CPU 和资源限制做线程数 sweep，分别找 prefill/decode 合适档位；不默认用满全部核心。
2. 验证缺 cpufreq 时分组与 CpuIds fallback，检查绑核集合和工作线程实际运行位置。
3. 分解 enqueue、barrier、spin/yield/block 成本，调任务粒度、连续输出分片和共享数据，测 false sharing。
4. 测持续有效内存带宽、cache miss、缺页、权重/KV 字节流；先减少重复拷贝与分配，再评估预取和大页等方案。
5. 追踪 STATIC/Eager 和 DYNAMIC/Defer 的复用、KV 扩容 fresh allocation、peak RSS 与冷启动 mmap/reorder。

主要修改位置：`CPURuntime.cpp`、`CPUBackend.cpp`、`ThreadPool.cpp`、相关 Execution、`source/core/BufferAllocator.cpp` 和 KV 管理器。
依赖：能拆分 compute/pack/调度/内存耗时，避免用线程调参掩盖未接通的核。
验收：单/多线程数值一致、绑核实测生效、多个新进程稳定；无明显冷启动、稳态或 RSS 回归。
可复用：`core/threadpool`、`core/buffer_allocator`、Attention 线程基准、模型 benchmark；记录持续带宽而非接口峰值。

### 第六步：精度、向量宽度和可选扩展

1. 在原 C4 ABI 下通过空间维、多行或多 panel 使用 RVV，再决定是否需要更宽 CUnit。
2. 动态 pack 或新权重布局属于独立工程：同步 Tensor 转换、GEMM 主/tail、卷积、Attention/KV 和 metadata。
3. SG2044 的 FP16/BF16 只有在 ISA、GCC 14 intrinsic/编译支持和完整函数表验证后才能列入实现。
4. Vec 抽象需要考虑标准 RVV sizeless 类型限制、C++11 和跨架构编译；禁止直接把 RVV 向量类型当普通结构体成员套用。
5. SpacemiT IME2/TCM 只保留为另一个硬件项目的可选路线，本轮 SG2044 不启用也不依赖它。

主要修改位置：`compute/CommonOptFunction.h/.cpp`、RVV kernel/packer、转换和所有相关消费者、`source/math/Vec.hpp`、RISC-V CMake。
依赖：前面阶段已有证据表明新契约或新精度值得投入，且原标量/标准 RVV fallback 独立可用。
验收：ISA/精度/pack/线程/tail 完整矩阵；新精度误差与模型质量；二进制大小、初始化和每次调用成本。
可复用：现有 MatMul、卷积、Convert、Norm、Attention 测试；新增只覆盖新契约和新分支的必要用例。

### 第七步：固定回归矩阵、CI 和体积验收

1. 将 SG2044/GCC 14 的标量与 RVV 两种构建、路径断言、op 正确性和真实板端阶段固化。
2. 现有 `test.sh local`、通用 Linux workflow 不等于已验证 RVV；接入显式 RVV 配置和目标硬件 runner。
3. 每个优化都保留原标量对照，最终覆盖代表性 CNN、Transformer/LLM；目标若含 Diffusion，再补对应模型。
4. 检查 shared/static 链接、安装接口、RVV OFF 和非 RISC-V 构建；统计 `.text/.rodata` 与峰值 RSS。
5. 不新增命名空间动态初始化对象；检查全局初始化符号，保持 C++11、无 RTTI/异常和性能/体积约束。

主要修改位置：`test/`、`test_stages.json`、`test.sh`、`.github/workflows/`、相关 CMake，按实际 runner 能力接入。
依赖：各阶段候选已完成板端验收，测试名、参数与产物版本可追溯。
验收：所有规定正确性阶段通过，性能收益超过噪声且无关键 shape 回归，体积/启动变化可解释；未验场景明确列出。

## 5. 每次优化必须采用的标量对照与板端门禁

用户要求每次优化都配 MNN 原有标量对照并在板端运行，采用以下两层对照：

- **函数级同进程对照**：保留 MNN 原标量实现，候选与原实现读取同一输入和实际 packed 数据，比较有效输出、padding、后处理和边界行为；不要只写另一份模拟候选算法的 oracle。
- **完整构建对照**：同一候选源码使用独立 scalar/RVV build 目录；RVV OFF 基线必须禁用编译器自动向量化，否则不能称为严格标量。另保留 `e44af4a6` RVV 产物比较实际升级收益。

严格标量 GCC 14 对照建议使用 `-march=rv64gc -mabi=lp64d -fno-tree-vectorize -fno-tree-slp-vectorize`；核对所有 TU 实际命令和最终汇编。
同进程基准应让原标量函数保留并独立编译为上述语义；不要给整份候选 RVV TU 禁用 ISA 后期望 intrinsic 仍可编译。
量化对照分阶段检查：packed 输入 → 整数 accumulator → scale/zero-point → bias/clamp → 输出布局。

| 门禁 | 必测内容 | 通过依据 |
|---|---|---|
| G0：编译与接入 | GCC 14、头文件签名、C/C++ linkage、符号唯一、RVV ON/OFF | 构建成功，实际入口命中，正确 fallback |
| G1：函数级 | 零/小/大长度、tile±1、非连续 stride、原位、padding、NaN/Inf（适用函数） | 与 MNN 原标量的既有数值契约一致 |
| G2：op 级 | 对应原测试、真实模型 shape、线程 1 和多线程、精度/量化分支 | SG2044 原测试与新增必要回归通过 |
| G3：模型级 | 固定采样短生成、长 prompt、跨 KV/pack 阈值、代表性模型 | 内容和精度正常，不仅是 benchmark 有速度输出 |
| G4：性能与体积 | 同进程交错 A/B；多新进程 A/B、B/A；冷/热、prefill/decode 分开 | 超过测量波动，模型收益可复现，RSS/体积变化记录完整 |

tile 边界至少覆盖 E=1、2、15、16、17；pack 覆盖 C/H/K 非整除；Attention 覆盖 63/64/65、2047/2048/2049、4095/4096/4097。
多 VLEN 可做额外仿真正确性验证；SG2044 真机只证明实测 VLEN，仿真耗时不作为真实硬件性能。
性能不设未经实测的承诺百分比。可记录 `speedup=baseline_time/candidate_time`，同时给 shape、线程与噪声范围。
decode 以每 token 实际权重/KV/metadata 字节数估算有效带宽，prefill 同时报 pack/量化和 GEMM 时间。

## 6. 板端构建模板与执行纪律

以下为原生 Linux riscv64 的待执行模板；使用已确认的 GCC 14 编译器路径填入占位符，实际命令和结果另行保存。

```bash
cmake -S . -B build-sg2044-rvv \
  -DCMAKE_C_COMPILER=<GCC14_C> -DCMAKE_CXX_COMPILER=<GCC14_CXX> \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMNN_USE_RVV=ON -DMNN_RVV_MARCH=rv64gcv \
  -DMNN_RVV_SPACEMIT_IME2=OFF -DMNN_RVV_FAST_MATH=OFF \
  -DMNN_BUILD_TEST=ON -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_USE_THREAD_POOL=ON -DMNN_OPENMP=OFF
cmake --build build-sg2044-rvv -j <JOBS>
```

独立 `build-sg2044-scalar` 保持其他选项一致，设置 `MNN_USE_RVV=OFF` 并为 C/C++ 添加上述严格标量参数。
若交叉编译，另用实际工具链文件声明 Linux/riscv64、GCC 14 和 sysroot；不得把宿主机编译成功当成板端验收。
测试前确认 `run_test.out` 的位置参数，核对输出中的 memory/thread/precision；只跑测试名不能证明目标配置。
必须确认匹配测试数 `passed > 0`；零匹配也可能返回成功。Attention 用例受 `MNN_SUPPORT_TRANSFORMER_FUSE` 控制，ThreadPool 用例也受对应编译宏控制。
低比特的 `MNN_LOW_MEMORY=ON` 是编译门，运行时还需第六个位置参数 `Memory_Low=2`，例如：

```bash
./build-sg2044-rvv/run_test.out op/convolution/weighti8i4conv2d 0 0 1 sg2044 2
```

上述命令仍需配实际内核命中证据，不能仅凭通过就断定 W4 专用内核已执行。
模型 profile 使用 `llm_bench --profile` 的真实选项，按 prefill/decode 分开；`llm_demo` 等内容验证补足合成 benchmark 的不足。

远端操作遵守 [RISC-V 板端验证规则](../skills/cpu/shared/riscv-remote-validation.md)：先只读检查远端分支、HEAD、工作区和 build，保留未提交修改，使用隔离目录或显式文件同步。
公开文档不保存服务器 IP、账号、SSH/代理配置、远端仓库/build/模型绝对路径。执行身份与连接信息仅作为会话参数。

## 7. 候选复用与后续结果记录

仓库存在 RVV ReLU-int8、HardSwish、Sigmoid、count/min/max、LinearAttention、量化 KV、Vec 和 channel-pack 候选分支；它们不等于已进入当前基线。
只能逐函数核对后复用：文本补丁可应用不代表接口正确、实际接通或板端更快。
已发现的复用风险包括量化 Value 补丁 `#ifndef` 范围过大、Vec 向量类型成员限制、channel-pack/Vec 重复修改和旧分支上下文冲突。
不要一次性合并全部候选；每一小批走同一套标量/板端门禁，数据支持后再保留。

| 工作项 | 基线状态 | 候选/后续状态 | SG2044 结果 |
|---|---|---|---|
| C4 pack/unpack、Scale、PReLU 接入 | linkage 问题待修 | 四项接线已实现；首轮目标对照已分析，下一版先改 Scale 连续访存 | Pack/Unpack 多数配置受益，Scale 广泛退化，PReLU 小尺寸退化；新 Scale 与完整库派发待验 |
| 全量算子覆盖与其他 linkage | 已做静态分类，尚非逐模型命中清单 | 按第一阶段继续 | 未取得 |
| E=1、W4/Fast、Softmax/C4 Norm | 发现明确候选缺口 | 等真实热点数据后依次实施 | 未取得 |
| pack/Execution/Attention/KV/线程 | 已列调用链、依赖和门禁 | 按第二至第五阶段执行 | 未取得 |
| 新精度/新 pack/CI 体积 | 专项计划 | 有收益证据和前置门禁后进入 | 未取得 |

后续每条完成记录应包含：候选提交、改动文件、原标量对照、板端测试集合、实际命中条件、性能适用范围、回退与未覆盖项。
本轮优先交付可验证的小批接入修复与 SG2044 数据，逐步扩展覆盖；“全部算子已优化”必须由覆盖表和实际测试共同证明。

## 8. 本轮实际完成的候选与验证

`fe7f86c6` 将四个已有 RVV 函数改为独立 `_RVV` 符号，并在 `supportRVV` 能力门内注册。
原通用函数、算法和 C4 布局保持原有契约。该提交修复接入问题，最终是否在 SG2044 各 shape 上更快仍需数据决定。

配套 [对比测试说明](../benchmark/rvv_scalar_compare/README.md) 和 [运行入口](../benchmark/rvv_scalar_compare/run.py)
已具备精确原标量源码提取、真实 RVV TU 编译、VLEN/hwprobe 探针、正确性检查和交错计时。
产物包含原始 JSONL/CSV 及每个 case 的中位数、MAD、`scalar_time / rvv_time`；零工作量不报告加速比。

| 已执行检查 | 结果 | 证据边界 |
|---|---|---|
| 四个生产 RVV TU 的 Clang 22 RISC-V C++11 编译、符号和签名检查 | 通过 | 不等于 GCC 14 编译或完整库链接通过 |
| 测试框架与原标量实现的本机检查 | 880 case 通过，无计时 | 验证提取与测试逻辑；此项未执行 RVV 指令 |
| 四个生产 RVV TU 的本机语义模拟 | VLEN=128/256/512/1024，各用 maximal/balanced 两种合法 vl 切分，共 8 组、10,120 次 case 执行通过，无计时 | 仅模拟语义；重复执行数不代表独立算子数或真机覆盖率 |
| 测试框架的参数留档、拒绝错误目标、失败保留、统计聚合 | 通过本机检查 | 不证明 SG2044 运行环境或函数表实际命中 |
| 目标环境预检（原始包已核验） | GCC 14.4.0，riscv64 target；标准库/线程程序通过；RVV FP32 探针通过，VLEN=128，hwprobe 报告 V | 固定 CPU 0，系统负载较高且温度记录缺失；不外推其他机器或完整 MNN 派发 |
| 四函数目标正确性（v2 原始数据） | 1320个参数配置，failed=0，无计时 | 原标量与当时的生产 RVV 函数直接对照，尚未检查完整 MNN 调用链 |
| 三个独立目标性能进程（v2 原始数据） | 每个1320配置、failed=0、18480条计时采样；源码/对象/二进制哈希一致，原始统计已复算 | 存在稳定退化，不能据passed认定性能验收；逐配置与分组结果由 analyze.py 输出 |
| 下一版 Scale 连续访存候选 | 本机多 VLEN 语义模拟、RISC-V 静态编译通过 | 修改后的实现尚无目标正确性或性能证据，不能沿用 v2 的通过状态 |
| 完整 MNN 构建、op/模型与实际派发验证 | 显式跳过 | integration=explicitly_skipped；仍须独立完成 |

本机语义模拟使用相同生产函数体和原标量函数体，覆盖尾通道、非连续 stride、原位 Scale/PReLU、
正负零、subnormal、Inf、NaN 与保护区。模拟不能替代目标执行；v2 原始数据已核验，新候选仍须目标执行。

首次目标预检因入口使用了未经验证的 512 位 VLEN 默认值而停止，实测值为 128 位；这不是四个内核的正确性失败。
入口已取消基于芯片名称的默认猜测，要求显式传入先前实测值。已有小包可用 `EXPECT_VLEN=128` 在同一目标复测，
每次探针仍须与预期一致，旧失败记录保留。四个 RVV 内核按运行时 VL 迭代，没有固定 512 位依赖。

第二次用户回传的目标运行通过 VLEN 预检，但旧框架在创建结果文件前退出。
该失败发生在测试框架入口，不能计作生产内核测试通过；随后 v2 的成功执行已取得原始日志和产物哈希。
已在本机复现参数校验不允许数字、从而拒绝 `target_riscv64_execution` 中 `64` 的问题；
修复为显式接受三个执行证据标签，并补齐提前退出的诊断信息。之前 host 标签不含数字，未覆盖该入口差异。
四个生产内核不需为此改动；随后 v2 包已回传上述函数级成功摘要。

本次1320是唯一参数配置数。正确性阶段及三个性能进程各执行同一矩阵，重复次数不能当作新配置或算子数量。
每个性能进程的18480采样等于1320×7轮×2实现，包含220个零工作量配置；这些配置不报告加速比。
结果包校验和、源码/编译器/对象身份和逐配置中位数/MAD已核验。原始数据与派生报告留在本地，
可用 `benchmark/rvv_scalar_compare/analyze.py` 从解包目录复算。

首轮结果决定以下近期顺序，不改变前文全库审计的依赖关系：

1. **Scale**：先修复跨尺寸的系统性退化。当前候选用连续 RVV load/store 和每组复用系数替代四路跨步访存；
   保留乘法后加法的语义，单 C4 直接计算。先做同矩阵目标复测，其他三个函数保持原版本作对照。
2. **PReLU**：处理小 area 与向量边界附近的退化，验证连续访存、掩码和原地语义。
3. **Pack/Unpack**：分别处理完整 C4 与尾通道、小 area 与非连续 stride；不能把尾通道收益外推到完整 C4。
4. **完整 MNN**：验证实际函数表派发、相关 op、短/长输入模型，再将真实热点带回第一阶段的 GEMM、Softmax、Norm 等任务。

绑定 CPU 不等于独占该核。后续复测应记录负载，在空闲时段重复；没有 PMU 或带宽证据前，
连续访存仅是待验证的优化方案，不能将内核退化完全归因于访存或系统负载。

板端提供 [一键验证入口](../benchmark/rvv_scalar_compare/sg2044_validate.sh) 与
[完整 MNN 集成回归](../benchmark/rvv_scalar_compare/integration.sh)。入口先保护原 checkout、核实 GCC 14 与
`<memory>`/RVV 探针，再跑函数级正确性、多个新进程性能对照及独立 scalar/RVV 构建。
构建与运行的日志、退出码和原始数据保留在新结果目录；v2 函数级阶段已完成原始数据分析，新候选与完整库阶段仍待执行。
可用 GDB 的无计时回归用于检查四个 RVV 函数的实际命中；缺少该证据时必须保留“派发未验证”状态。
运行方法见 [测试说明](../benchmark/rvv_scalar_compare/README.md#sg2044-validation-entrypoint)。

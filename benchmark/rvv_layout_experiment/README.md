# RISC-V FP32 布局探索

这是独立试验，不修改 MNN Tensor ABI、`core->pack`、生产算子或派发表。
首先回答 E=1 直接向量化与可复用权重预打包是否值得投入；再比较一条
MatMul → Scale → PReLU 链使用 P4/P8/P16 中间布局的成本。

## 已确认的调用关系

`CPUMatMul::_scheduleForVecE` 已直接调用 `MNNComputeMatMulForE_1`，E=1 不经过常规 A-pack。
因此不能把直接 RVV 版本称为“消除了现有 E=1 A-pack”。当前 packed remain E=1 则是另一条路径。
FP32 微测试结果不代表低比特卷积路径上的量化 LLM 性能。

## 布局与对照

* A 为 `[E,K]`；外部链输入为 C4，E>1 计入 C4 → 连续 A 转换。
* 原始 B 同时测试 `[K,H]` 和转置存储 `[H,K]`。
* 预打包 B 为 `[ceil(H/P),K,P]`，P 取 4、8、16，尾部补零。
* 中间 C 为 `[ceil(H/P),E,P]`；链输出恢复 C4，P4 无需转换。
* 所有 panel 用相同 m4 算法，VL 不超过当前 panel，区分布局变化与 LMUL 变化。
* `original_e1`：从指定源码精确提取原有标量函数，使用真实 `Vec.hpp`。
* `current_c4`：指定源码中的真实 RVV packed remain，仅测试 E=1。
* `direct_rvv`：原始 `[K,H]` 权重上的连续输出通道 RVV，完全不打包权重。
* `panel4/8/16`：实验性 FP32 panel kernel；Scale 与 PReLU 为两个独立向量化 pass。

E>1 的 `original_e1` 是重复调用原始 GEMV 的实验对照，**不是生产 GEMM**。
P8/P16 链应与同算法 P4 链比较，不能据此声称胜过 MNN 已有优化卷积或 GEMM。
固定使用 m4 仅用于控制变量；更优 LMUL 和多行 tile 尚未探索。

## 测量口径

1. 标量模拟必须通过，随后 native RVV 对独立 double oracle、权重重排、padding、保护区检查。
2. 23 个形状 × 两种权重方向 × 有/无偏置；包括 K/H 为零、尾部及大矩阵。
3. `compute` 不计权重打包；`weight_pack` 只计预分配缓冲中的打包。
4. `steady_chain` 计输入转换、计算、Scale、PReLU、输出转换。
5. `first_chain` 额外计每次 B 打包；不包括分配器、图调度或加载模型的时间。
6. 每个变体校准到至少约 1 ms，一轮内随机交错，7 轮 × 3 个独立进程，固定 CPU。
7. 每进程先取中位数，再汇总进程间比值；保留原始样本和实际迭代次数。
8. `amortization.json` 是由独立 pack/compute 测量推算的复用盈亏点，不是模型实测。

当前各方统一禁用 fast-math、FMA contraction、LTO；原始标量与控制/转换代码禁用自动向量化。
当前生产 C4 对照保留 GCC 自动向量化。运行前用独立 RVV FP32 probe 记录实际 VLEN。
转换器为正确性优先的标量原型，转换成本仍有优化空间。工作集重复使用，不模拟冷缓存或完整模型权重流。
没有多线程扩展、异常浮点专项或其他 VLEN 的生产验收，不能直接启用为全局后端布局。

## 运行

在空闲、支持标准 RVV 的 RISC-V 机器上，使用 GCC 14：

```bash
python3 benchmark/rvv_layout_experiment/run.py \
  --source /path/to/exact/MNN --out /path/to/new-results --cpu 8
python3 benchmark/rvv_layout_experiment/analyze.py \
  /path/to/new-results --out /path/to/new-analysis
```

输出目录必须不存在。运行脚本只读取显式列出的公开源码并保存其 SHA256，
保留编译命令、编译日志、标量模拟/native 正确性日志、三份 CSV、二进制和对象哈希。
先检查 `metadata.json` 的 `status`，再分析；错误运行不可作为性能证据。

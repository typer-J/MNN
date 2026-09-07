---
name: arm-cpu-optimize
description: ARM CPU 优化旧入口兼容页；流程已迁移到 skills/cpu，保留本地性能验证经验。
---

# ARM CPU 优化入口（兼容）

上游已将原流程整合进 [CPU skill](../cpu/SKILL.md)。性能分析请依次阅读
[优化入口](../cpu/optimize/SKILL.md) 和 [ARM 路径诊断](../cpu/optimize/arch/arm.md)；
已定位的内核实现请阅读 [kernel 入口](../cpu/kernel/SKILL.md)。
原 step0–step5 文件已移除，请按新入口的步骤及验收门禁执行。

## 本地保留的验证经验

7. **先建立已完成集合再审计缺口**：同时检查任务表、当前分支、上游提交和活跃功能分支，将候选分成“已合入”“分支已完成”“尚未开始”“已有实现但性能回退”。不要仅凭当前分支的文件名判断缺失，也不要把性能低于标量的 RVV 内核当作已完成
8. **成熟 SIMD 实现优先迁移**：当目标架构仍走标量路径时，优先选择已有 NEON/AVX/汇编实现的函数。先复用其数学拆分、循环展开、边界语义和测试，再做架构特有调优；没有可对照实现、需要重新设计算法的项目后排
9. **保持导出符号 linkage 一致**：若函数在 `CommonOptFunction.h` 的 `extern "C"` 区域声明，架构专用 `.cpp` 替换实现也必须显式 `extern "C"`；仅通过 `CoreFunctions` 内部指针注册的 `_RVV` / `_NEON` helper 才可保持普通 C++ linkage
10. **缺少目标真机时也要做最低静态验证**：每个架构专用新增文件至少执行目标编译器 `-fsyntax-only`（例如 RVV 用 `clang++ -target riscv64 -march=rv64gcv -mabi=lp64d`），再配合 `git diff --check`；这不能替代性能/正确性测试，但能及时发现 intrinsic 类型、头文件和 linkage 问题
11. **区分实现覆盖、微基准收益与端到端收益**：汇报或撰写 motivation 时，先用系统约束说明缺口，再分别列出“已有实现”“已测函数级数据”“尚缺模型级证据”。不得用已提交的 RVV kernel 数量代替性能结论，也不得把 microbenchmark speedup 外推为 session/module 端到端收益；低于 1x 的回归项应作为方法设计的直接证据和验收约束
12. **向量尾部必须按语义分类**：审计或优化可伸缩向量代码时，区分仅改变 active lane 数的 length tail、改变 pack/address mapping 的 layout-semantic tail，以及 reduction finalization。`setvl(remaining)` 只能统一第一类；不得用它声称消除了所有 remainder。tail 优化的因果消融应保持 full-vector body、输入和算法不变，只替换 scalar cleanup 与统一动态 `vl`
13. **跨 VLEN 主张要审计 ISA 与产物身份**：记录运行时 VLEN、RVV/SVE 版本或 profile、编译器、flags、source revision、源码 SHA-256 和 binary/object SHA-256。revision 不能识别未提交或未跟踪的源码差异；相同源码但不同产物只能证明 source portability。只有 ISA 编码和 ABI 兼容且二进制哈希相同，才能声称 same-binary portability。跨平台性能以各平台内部 speedup 和稳定性为主，不用不同服务器的绝对时间直接推导 ISA 优劣
14. **可伸缩向量类型要单独做目标语法检查**：RVV/SVE 的向量类型可能是 sizeless/scalable type，不能作为普通结构体成员、数组元素或通过取地址保存状态。辅助函数应按值接收/返回向量累加器，模板向量保留为函数局部值。即使宿主 fallback 已编译运行，也要用目标编译器对 intrinsic-only 路径执行 `-fsyntax-only`；缺少目标 C++ sysroot 时，建立只含目标 intrinsic 和原始指针的最小语法 harness，不能用宿主编译成功替代该检查
15. **真机脚本要预检 intrinsic API 而不只检查 ISA 宏**：`__riscv_vector` 只能说明编译参数启用了 V 扩展，不能证明当前 `riscv_vector.h` 提供代码所需的 intrinsic 版本。正式测量前，应使用最终的 `CXX`、`CXXFLAGS` 编译一个包含实际类型和函数（例如 `vfloat32m8_t`、`__riscv_vsetvlmax_e32m8`）的最小探针；同时记录 `command -v`、解析后的编译器路径和版本。冒烟与正式运行若分别使用 `c++` 和 `g++`，必须先确认二者解析为同一工具链，否则冒烟通过不能证明正式编译环境兼容

# RVV reduction and Int8 ReLU regression test

Run the direct-kernel test on a RISC-V Linux machine or under QEMU user emulation:

```sh
riscv64-linux-gnu-g++ -std=c++11 -O2 -march=rv64gcv -mabi=lp64d \
  -fno-fast-math -fno-tree-vectorize -fno-exceptions -fno-rtti -static \
  -DMNN_RVV_REDUCTION_RELU_TEST_MAIN -Iinclude -Isource -Ischema/current -I3rd_party/flatbuffers/include \
  test/backend/riscv/RVVReductionReluTest.cpp \
  source/backend/cpu/riscv/rvv/MNNCountMaxMinValue.cpp \
  source/backend/cpu/riscv/rvv/MNNReluInt8.cpp -o rvv-reduction-relu-test
qemu-riscv64 -cpu max,vlen=128 ./rvv-reduction-relu-test
qemu-riscv64 -cpu max,vlen=256 ./rvv-reduction-relu-test
qemu-riscv64 -cpu max,vlen=512 ./rvv-reduction-relu-test
```

The oracle follows the non-NEON scalar comparisons. Reduction cases include
lengths 0 through 257 and 4099, infinities, leading and later NaNs, signed zeros,
subnormals, and deterministic arbitrary float bit patterns. The empty-input case
checks the RVV helper's explicit zero result; the existing scalar helper requires
nonempty input. Results are compared bit for bit.

ReLU covers all representable zero points and out-of-range values from -256 to
255, lengths around vector boundaries, in-place operation, unaligned starting
addresses, and output guards. Its declaration comes from CommonOptFunction.h.

This source is inert in the regular test executable unless the standalone macro
is explicitly enabled. Emulation checks instruction-level correctness; it does
not establish performance on hardware or replace full MNN integration tests.

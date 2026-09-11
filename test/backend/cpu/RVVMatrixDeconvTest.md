# RVV matrix and depthwise deconvolution regression

```sh
$CXX -std=c++11 -O3 -march=rv64gcv -ffp-contract=off -pthread \
 -Iinclude -Isource -DMNN_RVV_MATRIX_TEST_MAIN \
 source/backend/cpu/riscv/rvv/MNNMatrixAdd.cpp \
 source/backend/cpu/riscv/rvv/MNNMatrixSub.cpp \
 source/backend/cpu/riscv/rvv/MNNMatrixProd.cpp \
 source/backend/cpu/riscv/rvv/MNNDeconvRunForUnitDepthWise.cpp \
 test/backend/cpu/RVVMatrixDeconvTest.cpp -o rvv-matrix-test
./rvv-matrix-test 128 1
./rvv-matrix-test 128 4
```

The first argument must match the measured VLEN. Repeat at VLEN 256/512/1024 under QEMU.
The reference checks exact scalar results, in-place matrix aliases, zero sizes, tails, padded
strides, overlapping deconvolution rows and unchanged guards. Also run MatMul and Deconvolution
operator tests with one and four threads. Compare performance with multiple fresh processes,
alternating A/B order; include single-row inputs and small contiguous filters.

#include "MNNRvvC4Functions.hpp"
#include <riscv_vector.h>

void MNNScaleAndAddBias_RVV(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                           size_t biasNumber) {
    if (planeNumber == 0 || biasNumber == 0) {
        return;
    }

    // A single C4 needs no coefficient replication.
    if (planeNumber == 1) {
        const size_t vl = __riscv_vsetvl_e32m1(4);
        for (size_t z = 0; z < biasNumber; ++z) {
            const vfloat32m1_t alphaC4 = __riscv_vle32_v_f32m1(alpha + 4 * z, vl);
            const vfloat32m1_t biasC4 = __riscv_vle32_v_f32m1(bias + 4 * z, vl);
            vfloat32m1_t data = __riscv_vle32_v_f32m1(src + 4 * z, vl);
            data = __riscv_vfmul_vv_f32m1(data, alphaC4, vl);
            data = __riscv_vfadd_vv_f32m1(data, biasC4, vl);
            __riscv_vse32_v_f32m1(dst + 4 * z, data, vl);
        }
        return;
    }

    const size_t maxVl = __riscv_vsetvlmax_e32m4();
    const vuint32m4_t channel = __riscv_vand_vx_u32m4(__riscv_vid_v_u32m4(maxVl), 3, maxVl);
    const size_t count = planeNumber * 4;
    for (size_t z = 0; z < biasNumber; ++z) {
        const vfloat32m1_t alphaC4 = __riscv_vle32_v_f32m1(alpha + 4 * z, 4);
        const vfloat32m1_t biasC4 = __riscv_vle32_v_f32m1(bias + 4 * z, 4);
        const vfloat32m4_t alphaRepeated = __riscv_vrgather_vv_f32m4(
            __riscv_vlmul_ext_v_f32m1_f32m4(alphaC4), channel, maxVl);
        const vfloat32m4_t biasRepeated = __riscv_vrgather_vv_f32m4(
            __riscv_vlmul_ext_v_f32m1_f32m4(biasC4), channel, maxVl);
        const float* srcZ = src + z * count;
        float* dstZ = dst + z * count;
        size_t remaining = count;
        while (remaining > 0) {
            // Bound AVL by VLMAX so VL stays a multiple of four, including the tail.
            const size_t vl = __riscv_vsetvl_e32m4(remaining < maxVl ? remaining : maxVl);
            vfloat32m4_t data = __riscv_vle32_v_f32m4(srcZ, vl);
            data = __riscv_vfmul_vv_f32m4(data, alphaRepeated, vl);
            data = __riscv_vfadd_vv_f32m4(data, biasRepeated, vl);
            __riscv_vse32_v_f32m4(dstZ, data, vl);
            srcZ += vl;
            dstZ += vl;
            remaining -= vl;
        }
    }
}

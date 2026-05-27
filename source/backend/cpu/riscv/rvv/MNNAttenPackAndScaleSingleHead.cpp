#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

void MNNAttenPackAndScaleSingleHead_RVV(float* dst, const float* srcHeadBase, size_t srcRowStride, const float* scale,
                                        const int32_t* units, size_t seqLen, size_t headDim) {
    const int32_t eP = units[0];
    const int32_t lP = units[1];
    const float scaleVal = scale[0];
    const size_t packedHeadDim = UP_DIV(headDim, lP);
    const size_t dstStrideDOuter = static_cast<size_t>(eP) * lP;
    const size_t dstStrideSOuter = packedHeadDim * dstStrideDOuter;
    const ptrdiff_t dstByteStride = static_cast<ptrdiff_t>(dstStrideDOuter * sizeof(float));

    for (size_t s = 0; s < seqLen; ++s) {
        const size_t sOuter = s / eP;
        const size_t sInner = s % eP;
        const float* srcRowPtr = srcHeadBase + s * srcRowStride;
        float* dstBasePtr = dst + sOuter * dstStrideSOuter + sInner * lP;

        size_t d = 0;
        while (d < headDim) {
            const size_t vl = __riscv_vsetvl_e32m8(headDim - d);
            vfloat32m8_t data = __riscv_vle32_v_f32m8(srcRowPtr + d, vl);
            data = __riscv_vfmul_vf_f32m8(data, scaleVal, vl);
            __riscv_vsse32_v_f32m8(dstBasePtr + d * dstStrideDOuter, dstByteStride, data, vl);
            d += vl;
        }
    }
}

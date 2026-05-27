#include <riscv_vector.h>
#include <cstddef>
#include <cstring>

void MNNFlashAttentionUpdateBlockOutput_RVV(float* dst, float* src, float* scale, float* normalizeScale, int depthQuad,
                                            int plane, int pack, int idx, int kvBlocks, int size, int bytes,
                                            int seqStart) {
    if (bytes != static_cast<int>(sizeof(float))) {
        if (idx > 0) {
            for (int j = 0; j < depthQuad; ++j) {
                const int stride0 = plane * pack;
                for (int i = seqStart; i < plane; ++i) {
                    for (int k = 0; k < pack; ++k) {
                        const int offset = j * stride0 + i * pack + k;
                        dst[offset] = src[offset] + dst[offset] * scale[i];
                    }
                }
            }
        } else {
            std::memcpy(dst, src, static_cast<size_t>(size) * bytes);
        }
        if (idx == kvBlocks - 1) {
            const int stride0 = plane * pack;
            for (int j = 0; j < depthQuad; ++j) {
                for (int i = 0; i < plane; ++i) {
                    const float normalize = 1.0f / normalizeScale[i];
                    for (int k = 0; k < pack; ++k) {
                        dst[j * stride0 + i * pack + k] *= normalize;
                    }
                }
            }
        }
        return;
    }

    const int stride0 = plane * pack;
    if (idx > 0) {
        for (int j = 0; j < depthQuad; ++j) {
            for (int i = seqStart; i < plane; ++i) {
                int remain = pack;
                float* dstPtr = dst + j * stride0 + i * pack;
                float* srcPtr = src + j * stride0 + i * pack;
                while (remain > 0) {
                    const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(remain));
                    vfloat32m8_t newData = __riscv_vle32_v_f32m8(srcPtr, vl);
                    const vfloat32m8_t oldData = __riscv_vle32_v_f32m8(dstPtr, vl);
                    newData = __riscv_vfmacc_vf_f32m8(newData, scale[i], oldData, vl);
                    __riscv_vse32_v_f32m8(dstPtr, newData, vl);
                    dstPtr += vl;
                    srcPtr += vl;
                    remain -= static_cast<int>(vl);
                }
            }
        }
    } else {
        std::memcpy(dst, src, static_cast<size_t>(size) * bytes);
    }

    if (idx == kvBlocks - 1) {
        for (int j = 0; j < depthQuad; ++j) {
            for (int i = 0; i < plane; ++i) {
                int remain = pack;
                float* dstPtr = dst + j * stride0 + i * pack;
                const float normalize = 1.0f / normalizeScale[i];
                while (remain > 0) {
                    const size_t vl = __riscv_vsetvl_e32m8(static_cast<size_t>(remain));
                    vfloat32m8_t data = __riscv_vle32_v_f32m8(dstPtr, vl);
                    data = __riscv_vfmul_vf_f32m8(data, normalize, vl);
                    __riscv_vse32_v_f32m8(dstPtr, data, vl);
                    dstPtr += vl;
                    remain -= static_cast<int>(vl);
                }
            }
        }
    }
}

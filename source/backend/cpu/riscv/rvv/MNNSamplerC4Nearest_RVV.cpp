#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

namespace MNN {
namespace CV {
struct Point;
} // namespace CV
} // namespace MNN

static inline float clampSamplerCoordinate(float value, float maxValue) {
    if (value < 0.0f) {
        return 0.0f;
    }
    return value > maxValue ? maxValue : value;
}

void MNNSamplerC4Nearest_RVV(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                             size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    (void)capacity;
    dest += 4 * sta;
    const float* pointData = reinterpret_cast<const float*>(points);
    float currentX = pointData[0];
    float currentY = pointData[1];
    const float dx = pointData[2];
    const float dy = pointData[3];
    const float xMax = (float)(iw - 1);
    const float yMax = (float)(ih - 1);

    // The gather index is 32-bit, so the batch is capped at 16 lanes. Capping by
    // vlmax as well keeps `vl == request` on small-VLEN cores (VLEN=64 gives
    // only 8 lanes for e32m4); advancing by `request` there would skip pixels.
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    const size_t chunk = vlmax < 16 ? vlmax : 16;
    size_t i = 0;
    while (i < count) {
        const size_t remaining = count - i;
        const size_t request = remaining < chunk ? remaining : chunk;
        const size_t vl = __riscv_vsetvl_e32m4(request);
        uint32_t offsets[16];
        size_t wideOffsets[16];
        bool fitsU32 = true;
        for (size_t lane = 0; lane < vl; ++lane) {
            const int y = (int)__builtin_roundf(clampSamplerCoordinate(currentY, yMax));
            const int x = (int)__builtin_roundf(clampSamplerCoordinate(currentX, xMax));
            const size_t sourceOffset = (size_t)y * yStride + 4 * (size_t)x;
            wideOffsets[lane] = sourceOffset;
            if (sourceOffset > UINT32_MAX - 3) {
                fitsU32 = false;
            }
            offsets[lane] = (uint32_t)sourceOffset;
            currentY += dy;
            currentX += dx;
        }

        if (fitsU32) {
            const vuint32m4_t index = __riscv_vle32_v_u32m4(offsets, vl);
            const vuint32m4_t pixels = __riscv_vluxei32_v_u32m4((const uint32_t*)source, index, vl);
            __riscv_vse32_v_u32m4((uint32_t*)(dest + 4 * i), pixels, vl);
        } else {
            // Offsets past 4GiB cannot be expressed as a 32-bit gather index.
            // Only mixed architectures build such tensors; fall back per lane.
            for (size_t lane = 0; lane < vl; ++lane) {
                const unsigned char* src = source + wideOffsets[lane];
                unsigned char* dst = dest + 4 * (i + lane);
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        }
        i += vl;
    }
}

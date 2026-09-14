#include <riscv_vector.h>

void MNNC3ToFloatC3_RVV(const unsigned char* source, float* dest, const float* mean, const float* normal,
                        size_t count) {
    size_t i = 0;
    while (i < count) {
        const size_t vl = __riscv_vsetvl_e8m2(count - i);
        for (size_t c = 0; c < 3; ++c) {
            const vuint8m2_t sourceValue = __riscv_vlse8_v_u8m2(source + 3 * i + c, 3, vl);
            const vuint32m8_t sourceWide = __riscv_vzext_vf4_u32m8(sourceValue, vl);
            vfloat32m8_t value = __riscv_vfcvt_f_xu_v_f32m8(sourceWide, vl);
            value = __riscv_vfsub_vf_f32m8(value, mean[c], vl);
            value = __riscv_vfmul_vf_f32m8(value, normal[c], vl);
            __riscv_vsse32_v_f32m8(dest + 3 * i + c, 3 * sizeof(float), value, vl);
        }
        i += vl;
    }
}

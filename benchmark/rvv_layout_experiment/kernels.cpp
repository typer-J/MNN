// Experimental FP32 panel layouts; not registered in MNN CoreFunctions.
#include <algorithm>
#include <cstddef>
#ifndef EMULATE
#include <riscv_vector.h>
#endif

// B: [ceil(H/P), K, P], C: [ceil(H/P), E, P], A: [E,K].
// Fixed panel P is independent of hardware VL. Every tail is bounded by P.
extern "C" void panel(const float* a, const float* b, float* c, const float* bias, int e, int k, int h, int p) {
    for (int hb = 0; hb < (h + p - 1) / p; ++hb) {
        for (int row = 0; row < e; ++row) {
            for (int off = 0; off < p;) {
#ifdef EMULATE
                int vl = std::min(p - off, 16);
                float acc[16] = {};
                for (int i = 0; i < vl; ++i)
                    acc[i] = bias && hb * p + off + i < h ? bias[hb * p + off + i] : 0.0f;
                for (int z = 0; z < k; ++z)
                    for (int i = 0; i < vl; ++i)
                        acc[i] += a[row * k + z] * b[(hb * k + z) * p + off + i];
                for (int i = 0; i < vl; ++i)
                    c[(hb * e + row) * p + off + i] = acc[i];
#else
                size_t vl = __riscv_vsetvl_e32m4(p - off);
                auto acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                int valid = std::max(0, std::min(int(vl), h - hb * p - off));
                if (bias && valid) {
                    // Bias is copied into a zero-padded temporary by the caller.
                    acc = __riscv_vle32_v_f32m4(bias + hb * p + off, vl);
                }
                for (int z = 0; z < k; ++z) {
                    auto w = __riscv_vle32_v_f32m4(b + (hb * k + z) * p + off, vl);
                    acc = __riscv_vfadd_vv_f32m4(acc, __riscv_vfmul_vf_f32m4(w, a[row * k + z], vl), vl);
                }
                __riscv_vse32_v_f32m4(c + (hb * e + row) * p + off, acc, vl);
#endif
                off += vl;
            }
        }
    }
}

// Raw B [K,H], vectorizing independent output channels, no weight packing.
extern "C" void direct(const float* a, const float* b, float* c, const float* bias, int e, int k, int h) {
    for (int row = 0; row < e; ++row) {
        for (int y = 0; y < h;) {
#ifdef EMULATE
            int vl = std::min(h - y, 16);
            float acc[16] = {};
            for (int i = 0; i < vl; ++i)
                acc[i] = bias ? bias[y + i] : 0;
            for (int z = 0; z < k; ++z)
                for (int i = 0; i < vl; ++i)
                    acc[i] += a[row * k + z] * b[z * h + y + i];
            for (int i = 0; i < vl; ++i)
                c[row * h + y + i] = acc[i];
#else
            size_t vl = __riscv_vsetvl_e32m4(h - y);
            auto acc = bias ? __riscv_vle32_v_f32m4(bias + y, vl) : __riscv_vfmv_v_f_f32m4(0.0f, vl);
            for (int z = 0; z < k; ++z) {
                auto w = __riscv_vle32_v_f32m4(b + z * h + y, vl);
                acc = __riscv_vfadd_vv_f32m4(acc, __riscv_vfmul_vf_f32m4(w, a[row * k + z], vl), vl);
            }
            __riscv_vse32_v_f32m4(c + row * h + y, acc, vl);
#endif
            y += vl;
        }
    }
}

// Separate Scale and PReLU passes over a panel output; same algorithm for all P.
extern "C" void post(float* c, const float* scale, const float* bias, const float* slope, int e, int h, int p) {
    for (int pass = 0; pass < 2; ++pass)
        for (int hb = 0; hb < (h + p - 1) / p; ++hb)
            for (int row = 0; row < e; ++row)
                for (int off = 0; off < p;) {
                    float* dst = c + (hb * e + row) * p + off;
                    int ch = hb * p + off;
#ifdef EMULATE
                    int vl = std::min(p - off, 16);
                    for (int i = 0; i < vl; ++i) {
                        if (!pass)
                            dst[i] = dst[i] * scale[ch + i] + bias[ch + i];
                        else if (dst[i] < 0)
                            dst[i] *= slope[ch + i];
                    }
#else
                    size_t vl = __riscv_vsetvl_e32m4(p - off);
                    auto x = __riscv_vle32_v_f32m4(dst, vl);
                    if (!pass) {
                        x = __riscv_vfmul_vv_f32m4(x, __riscv_vle32_v_f32m4(scale + ch, vl), vl);
                        x = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(bias + ch, vl), vl);
                    } else {
                        auto negative = __riscv_vmflt_vf_f32m4_b8(x, 0.0f, vl);
                        auto product = __riscv_vfmul_vv_f32m4(x, __riscv_vle32_v_f32m4(slope + ch, vl), vl);
                        x = __riscv_vmerge_vvm_f32m4(x, product, negative, vl);
                    }
                    __riscv_vse32_v_f32m4(dst, x, vl);
#endif
                    off += vl;
                }
}

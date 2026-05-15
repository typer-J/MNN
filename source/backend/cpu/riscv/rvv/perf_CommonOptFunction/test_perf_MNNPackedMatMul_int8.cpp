#include "perf_common.h"

#include "../MNNPackedMatMul_int8.cpp"

static void MNNPackedMatMulRemain_int8_scalar_impl(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride, const float* k, const float* b) {
    const int8_t* B = reinterpret_cast<const int8_t*>(fB);
    const size_t h = parameter[2];
    const size_t l = parameter[1];
    const size_t cStride = parameter[3] / sizeof(float);
    const size_t bStride = parameter[5] + 4 * l;
    const size_t hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::max();
    if (postParameters != nullptr) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    const int blockId = static_cast<int>(parameter[6]);
    for (size_t x = 0; x < eSize; ++x) {
        float* dst = C + 4 * x;
        const float* src = A + x;
        for (size_t y = 0; y < hC4; ++y) {
            float* dstY = dst + y * cStride;
            const int8_t* weight = B + y * bStride;
            const float* alpha = k + y * 4;
            const float* qbias = b + y * 4;
            float summer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (blockId > 0) {
                std::memcpy(summer, dstY, 4 * sizeof(float));
            }
            if (bias != nullptr && postParameters != nullptr) {
                for (int v = 0; v < 4; ++v) {
                    summer[v] += bias[4 * y + v];
                }
            }
            for (size_t z = 0; z < l; ++z) {
                const float aVal = src[z * aStride];
                const int8_t* w = weight + z * 4;
                for (int v = 0; v < 4; ++v) {
                    summer[v] += (static_cast<float>(w[v]) * alpha[v] + qbias[v]) * aVal;
                }
            }
            for (int v = 0; v < 4; ++v) {
                dstY[v] = std::max(minValue, std::min(maxValue, summer[v]));
            }
        }
    }
}

static void MNNPackedMatMul_int8_scalar(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    MNNPackedMatMulRemain_int8_scalar_impl(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}

#ifndef PERF_MATMUL_HELPERS_ONLY
int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t L = 256;
    constexpr size_t H = 128;
    constexpr size_t eSize = 16;
    constexpr int warmup = 50;
    constexpr int loop = 3000;
    size_t parameter[7] = {0, L, H, H * 4 * sizeof(float), 0, 0, 0};
    float postParams[4] = {0.0f, 0.0f, -10000.0f, 10000.0f};
    const size_t hC4 = UP_DIV(H, 4);
    const size_t ASize = eSize * L;
    const size_t BSize = hC4 * (4 * L + parameter[5]);
    const size_t CSize = hC4 * (parameter[3] / sizeof(float));

    uint64_t seed = 0x9a700001ULL;
    std::vector<float> A(ASize), ref(CSize), out(CSize), bias(hC4 * 4), k(hC4 * 4), b(hC4 * 4);
    std::vector<int8_t> B(BSize);
    for (auto& x : A) x = perf_uniform_float(seed, -1.0f, 1.0f);
    for (auto& x : B) x = perf_uniform_i8(seed);
    for (auto& x : bias) x = perf_uniform_float(seed, -1.0f, 1.0f);
    for (auto& x : k) x = perf_uniform_float(seed, 0.001f, 0.05f);
    for (auto& x : b) x = perf_uniform_float(seed, -0.2f, 0.2f);

    std::fill(ref.begin(), ref.end(), 0.0f);
    std::fill(out.begin(), out.end(), 0.0f);
    MNNPackedMatMul_int8_scalar(ref.data(), A.data(), reinterpret_cast<const float*>(B.data()), parameter, postParams, bias.data(), k.data(), b.data());
    MNNPackedMatMul_int8(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), parameter, postParams, bias.data(), k.data(), b.data());
    for (size_t i = 0; i < CSize; ++i) {
        if (!perf_close(ref[i], out[i], 1e-3f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNPackedMatMul_int8_scalar : MNNPackedMatMul_int8;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), parameter, postParams, bias.data(), k.data(), b.data());
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), parameter, postParams, bias.data(), k.data(), b.data());
    }
    perf_touch(out.data(), out.size());
    return 0;
}
#endif

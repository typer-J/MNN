#include "perf_common.h"

#include "../MNNConvInt8ComputeBiasFloat.cpp"

static void MNNConvInt8ComputeBiasFloat_scalar(float* dst, const int32_t* bias, const float* weightScale,
                                               float scaleRatio, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<float>(bias[i]) * weightScale[i] * scaleRatio;
    }
}

static bool verify_bias_float_case(size_t size, float scaleRatio, uint64_t& seed) {
    std::vector<int32_t> bias(size);
    std::vector<float> weightScale(size), ref(size), out(size);
    for (size_t i = 0; i < size; ++i) {
        bias[i] = perf_uniform_int(seed, -100000, 100000);
        weightScale[i] = perf_uniform_float(seed, 0.0001f, 0.01f);
    }
    MNNConvInt8ComputeBiasFloat_scalar(ref.data(), bias.data(), weightScale.data(), scaleRatio, size);
    MNNConvInt8ComputeBiasFloat_RVV(out.data(), bias.data(), weightScale.data(), scaleRatio, size);
    for (size_t i = 0; i < size; ++i) {
        if (!perf_close(ref[i], out[i], 1e-4f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    uint64_t seed = 0xa77e4001ULL;
    if (!verify_bias_float_case(1, 1.0f, seed) || !verify_bias_float_case(31, 0.125f, seed) ||
        !verify_bias_float_case(1025, 2.25f, seed)) {
        return 1;
    }

    constexpr size_t size = 1 << 20;
    constexpr int warmup = 50;
    constexpr int loop = 3000;
    constexpr float scaleRatio = 0.125f;
    std::vector<int32_t> bias(size);
    std::vector<float> weightScale(size), dst(size);
    for (size_t i = 0; i < size; ++i) {
        bias[i] = perf_uniform_int(seed, -100000, 100000);
        weightScale[i] = perf_uniform_float(seed, 0.0001f, 0.01f);
    }

    auto fn = mode == 0 ? MNNConvInt8ComputeBiasFloat_scalar : MNNConvInt8ComputeBiasFloat_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(dst.data(), bias.data(), weightScale.data(), scaleRatio, size);
    }
    for (int i = 0; i < loop; ++i) {
        fn(dst.data(), bias.data(), weightScale.data(), scaleRatio, size);
    }
    perf_touch(dst.data(), dst.size());
    return 0;
}

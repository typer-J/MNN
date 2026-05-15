#include "perf_common.h"

#include "../MNNAbsMaxFP32.cpp"

static void MNNAbsMaxFP32_scalar(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    const size_t srcStep = static_cast<size_t>(pack) * realSize;
    for (size_t i = 0; i < realSize; ++i) {
        float absmaxVal = 0.0f;
        for (size_t c = 0; c < src_depth_quad; ++c) {
            const float* src = source + c * srcStep + i * pack;
            for (int k = 0; k < pack; ++k) {
                absmaxVal = std::max(absmaxVal, std::fabs(src[k]));
            }
        }
        absmax[i] = absmaxVal;
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t depth = 128;
    constexpr size_t realSize = 1024;
    constexpr int pack = 16;
    constexpr int warmup = 50;
    constexpr int loop = 3000;
    const size_t total = depth * realSize * pack;

    uint64_t seed = 0xab5a0001ULL;
    std::vector<float> src(total);
    std::vector<float> ref(realSize), out(realSize);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -100.0f, 100.0f);
    }

    MNNAbsMaxFP32_scalar(src.data(), ref.data(), depth, realSize, pack);
    MNNAbsMaxFP32(src.data(), out.data(), depth, realSize, pack);
    for (size_t i = 0; i < realSize; ++i) {
        if (!perf_close(ref[i], out[i], 1e-5f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNAbsMaxFP32_scalar : MNNAbsMaxFP32;
    for (int i = 0; i < warmup; ++i) {
        fn(src.data(), out.data(), depth, realSize, pack);
    }
    for (int i = 0; i < loop; ++i) {
        fn(src.data(), out.data(), depth, realSize, pack);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

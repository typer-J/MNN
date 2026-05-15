#include "perf_common.h"

#include "../MNNDynamicQuantFP32.cpp"

static void MNNDynamicQuantFP32_scalar(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias = nullptr) {
    const size_t stride = static_cast<size_t>(pack) * realSize;
    for (size_t i = 0; i < realSize; ++i) {
        const float scaleVal = scale[i];
        const float biasVal = bias == nullptr ? 0.0f : bias[i];
        for (size_t c = 0; c < src_depth_quad; ++c) {
            const float* srcZ = src + c * stride + i * pack;
            int8_t* dstZ = dst + c * stride + i * pack;
            for (int k = 0; k < pack; ++k) {
                int val = static_cast<int>(std::round(srcZ[k] * scaleVal + biasVal));
                val = std::max(-128, std::min(127, val));
                dstZ[k] = static_cast<int8_t>(val);
            }
        }
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

    uint64_t seed = 0xd7000001ULL;
    std::vector<float> src(total), scale(realSize), bias(realSize);
    std::vector<int8_t> ref(total), out(total);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -8.0f, 8.0f);
    }
    for (size_t i = 0; i < realSize; ++i) {
        scale[i] = perf_uniform_float(seed, 8.0f, 24.0f);
        bias[i] = perf_uniform_float(seed, -4.0f, 4.0f);
    }

    MNNDynamicQuantFP32_scalar(src.data(), ref.data(), scale.data(), depth, realSize, pack, bias.data());
    MNNDynamicQuantFP32(src.data(), out.data(), scale.data(), depth, realSize, pack, bias.data());
    for (size_t i = 0; i < total; ++i) {
        if (ref[i] != out[i]) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%d rvv=%d\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNDynamicQuantFP32_scalar : MNNDynamicQuantFP32;
    for (int i = 0; i < warmup; ++i) {
        fn(src.data(), out.data(), scale.data(), depth, realSize, pack, bias.data());
    }
    for (int i = 0; i < loop; ++i) {
        fn(src.data(), out.data(), scale.data(), depth, realSize, pack, bias.data());
    }
    perf_touch(out.data(), out.size());
    return 0;
}

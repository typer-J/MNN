#include "perf_common.h"

#include "../MNNFloat2Int8.cpp"

static void MNNFloat2Int8_scalar(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue,
                                 ssize_t maxValue, const float* zeroPoint, ssize_t quanParamVec) {
    float scale4[4] = {scalep[0], scalep[0], scalep[0], scalep[0]};
    float zero4[4] = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};
    if (quanParamVec & 1) {
        std::memcpy(scale4, scalep, 4 * sizeof(float));
    }
    if (quanParamVec >> 1) {
        std::memcpy(zero4, zeroPoint, 4 * sizeof(float));
    }
    for (size_t i = 0; i < sizeQuad; ++i) {
        for (int j = 0; j < 4; ++j) {
            int v = static_cast<int>(std::round(src[4 * i + j] * scale4[j] + zero4[j]));
            v = perf_clamp_i32(v, static_cast<int32_t>(minValue), static_cast<int32_t>(maxValue));
            dst[4 * i + j] = static_cast<int8_t>(v);
        }
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t sizeQuad = 262144;
    constexpr size_t total = sizeQuad * 4;
    constexpr int warmup = 50;
    constexpr int loop = 3000;
    float scale[4] = {6.25f, 6.25f, 6.25f, 6.25f};
    float zeroPoint[4] = {1.7f, 1.7f, 1.7f, 1.7f};
    constexpr ssize_t minValue = -127;
    constexpr ssize_t maxValue = 127;
    constexpr ssize_t quanParamVec = 0;

    uint64_t seed = 0xf2080001ULL;
    std::vector<float> src(total);
    std::vector<int8_t> ref(total), out(total);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -20.0f, 20.0f);
    }

    MNNFloat2Int8_scalar(src.data(), ref.data(), sizeQuad, scale, minValue, maxValue, zeroPoint, quanParamVec);
    MNNFloat2Int8_RVV(src.data(), out.data(), sizeQuad, scale, minValue, maxValue, zeroPoint, quanParamVec);
    for (size_t i = 0; i < total; ++i) {
        if (std::abs(static_cast<int>(ref[i]) - static_cast<int>(out[i])) > 1) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%d rvv=%d\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNFloat2Int8_scalar : MNNFloat2Int8_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(src.data(), out.data(), sizeQuad, scale, minValue, maxValue, zeroPoint, quanParamVec);
    }
    for (int i = 0; i < loop; ++i) {
        fn(src.data(), out.data(), sizeQuad, scale, minValue, maxValue, zeroPoint, quanParamVec);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

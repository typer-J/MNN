#include "perf_common.h"

#include "../MNNInt8ScaleToFloat.cpp"

static void MNNInt8ScaleToFloat_scalar(float* dst, const int8_t* src, const float* scale, size_t size,
                                       const float* zeroPoint, ssize_t quantParamVec) {
    float scale4[4] = {scale[0], scale[0], scale[0], scale[0]};
    float zero4[4] = {zeroPoint[0], zeroPoint[0], zeroPoint[0], zeroPoint[0]};
    if (quantParamVec & 1) {
        std::memcpy(scale4, scale, 4 * sizeof(float));
    }
    if (quantParamVec >> 1) {
        std::memcpy(zero4, zeroPoint, 4 * sizeof(float));
    }
    for (size_t i = 0; i < size; ++i) {
        for (int j = 0; j < 4; ++j) {
            dst[4 * i + j] = static_cast<float>(src[4 * i + j] - zero4[j]) * scale4[j];
        }
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t size = 262144;
    constexpr size_t total = size * 4;
    constexpr int warmup = 50;
    constexpr int loop = 3000;
    float scale[4] = {0.015625f, 0.015625f, 0.015625f, 0.015625f};
    float zeroPoint[4] = {-3.0f, -3.0f, -3.0f, -3.0f};
    constexpr ssize_t quantParamVec = 0;

    uint64_t seed = 0x51080001ULL;
    std::vector<int8_t> src(total);
    std::vector<float> ref(total), out(total);
    for (auto& x : src) {
        x = perf_uniform_i8(seed);
    }

    MNNInt8ScaleToFloat_scalar(ref.data(), src.data(), scale, size, zeroPoint, quantParamVec);
    MNNInt8ScaleToFloat_RVV(out.data(), src.data(), scale, size, zeroPoint, quantParamVec);
    for (size_t i = 0; i < total; ++i) {
        if (!perf_close(ref[i], out[i], 1e-6f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNInt8ScaleToFloat_scalar : MNNInt8ScaleToFloat_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), scale, size, zeroPoint, quantParamVec);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), scale, size, zeroPoint, quantParamVec);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

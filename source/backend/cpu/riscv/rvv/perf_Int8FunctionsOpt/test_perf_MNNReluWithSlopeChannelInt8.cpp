#include "perf_common.h"

typedef struct {
    float* inputScale;
    float* outputScale;
    int32_t* inputZeroPoint;
    int32_t* outputZeroPoint;
    int32_t minValue;
    int32_t maxValue;
} QuanPrePostParameters;

#include "../MNNReluWithSlopeChannelInt8.cpp"

static void MNNReluWithSlopeChannelInt8_scalar(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber,
                                               size_t depthQuad, const QuanPrePostParameters* params, size_t pack) {
    const float inputZero = static_cast<float>(params->inputZeroPoint[0]);
    const float outputZero = static_cast<float>(params->outputZeroPoint[0]);
    const float inputScale = params->inputScale[0];
    const float outputScale = params->outputScale[0];
    const int32_t minValue = params->minValue;
    const int32_t maxValue = params->maxValue;
    for (size_t j = 0; j < depthQuad; ++j) {
        const float* slopeZ = slope + pack * j;
        const int8_t* srcZ = src + pack * j * planeNumber;
        int8_t* dstZ = dst + pack * j * planeNumber;
        for (size_t i = 0; i < planeNumber; ++i) {
            const int8_t* srcX = srcZ + pack * i;
            int8_t* dstX = dstZ + pack * i;
            for (size_t c = 0; c < pack; ++c) {
                float value = (static_cast<float>(srcX[c]) - inputZero) * inputScale;
                if (value < 0.0f) {
                    value *= slopeZ[c];
                }
                int out = static_cast<int>(std::round(value * outputScale + outputZero));
                out = perf_clamp_i32(out, minValue, maxValue);
                dstX[c] = static_cast<int8_t>(out);
            }
        }
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t planeNumber = 2048;
    constexpr size_t depthQuad = 16;
    constexpr size_t pack = 16;
    constexpr size_t total = planeNumber * depthQuad * pack;
    constexpr int warmup = 50;
    constexpr int loop = 2000;

    uint64_t seed = 0x52180001ULL;
    std::vector<int8_t> src(total), ref(total), out(total);
    std::vector<float> slope(depthQuad * pack);
    for (auto& x : src) {
        x = perf_uniform_i8(seed, -80, 80);
    }
    for (auto& x : slope) {
        x = perf_uniform_float(seed, 0.02f, 0.25f);
    }

    float inputScale[1] = {0.2f};
    float outputScale[1] = {5.0f};
    int32_t inputZeroPoint[1] = {0};
    int32_t outputZeroPoint[1] = {0};
    QuanPrePostParameters params = {inputScale, outputScale, inputZeroPoint, outputZeroPoint, -127, 127};

    MNNReluWithSlopeChannelInt8_scalar(ref.data(), src.data(), slope.data(), planeNumber, depthQuad, &params, pack);
    MNNReluWithSlopeChannelInt8_RVV(out.data(), src.data(), slope.data(), planeNumber, depthQuad, &params, pack);
    for (size_t i = 0; i < total; ++i) {
        if (std::abs(static_cast<int>(ref[i]) - static_cast<int>(out[i])) > 1) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%d rvv=%d\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNReluWithSlopeChannelInt8_scalar : MNNReluWithSlopeChannelInt8_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), slope.data(), planeNumber, depthQuad, &params, pack);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), slope.data(), planeNumber, depthQuad, &params, pack);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

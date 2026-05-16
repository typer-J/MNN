#include "perf_common.h"

#ifndef MNN_RVV_QUAN_POST_TREAT_PARAMETERS_DEFINED
#define MNN_RVV_QUAN_POST_TREAT_PARAMETERS_DEFINED
typedef struct {
    const float* bias;
    const float* scale;
    int32_t minValue;
    int32_t maxValue;
} QuanPostTreatParameters;
#endif

#include "../MNNLineDepthWiseInt8AddBiasScaleUnit.cpp"

static void MNNLineDepthWiseInt8AddBiasScaleUnit_scalar(int8_t* dst, const int8_t* src, const int8_t* weight,
                                                        const QuanPostTreatParameters* parameters, size_t width,
                                                        size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
                                                        size_t dilateY_step, int8_t* idxOrder) {
    (void)idxOrder;
    constexpr size_t pack = 16;
    for (size_t dx = 0; dx < width; ++dx) {
        int32_t sum[pack] = {0};
        const int8_t* srcZ = src + src_w_step * dx;
        for (size_t fy = 0; fy < fh; ++fy) {
            const int8_t* srcY = srcZ + fy * dilateY_step;
            const int8_t* weightY = weight + fy * fw * pack;
            for (size_t fx = 0; fx < fw; ++fx) {
                const int8_t* srcX = srcY + fx * dilateX_step;
                const int8_t* weightX = weightY + fx * pack;
                for (size_t c = 0; c < pack; ++c) {
                    sum[c] += static_cast<int32_t>(srcX[c]) * static_cast<int32_t>(weightX[c]);
                }
            }
        }
        int8_t* dstX = dst + dx * pack;
        for (size_t c = 0; c < pack; ++c) {
            int value =
                static_cast<int>(std::round((static_cast<float>(sum[c]) + parameters->bias[c]) * parameters->scale[c]));
            value = perf_clamp_i32(value, parameters->minValue, parameters->maxValue);
            dstX[c] = static_cast<int8_t>(value);
        }
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t width = 512;
    constexpr size_t fw = 3;
    constexpr size_t fh = 3;
    constexpr size_t pack = 16;
    constexpr size_t inputWidth = width + fw - 1;
    constexpr size_t src_w_step = pack;
    constexpr size_t dilateX_step = pack;
    constexpr size_t dilateY_step = inputWidth * pack;
    constexpr int warmup = 50;
    constexpr int loop = 5000;
    const size_t srcSize = inputWidth * fh * pack;
    const size_t weightSize = fw * fh * pack;
    const size_t dstSize = width * pack;

    uint64_t seed = 0xd1f80001ULL;
    std::vector<int8_t> src(srcSize), weight(weightSize), ref(dstSize), out(dstSize);
    std::vector<float> bias(pack), scale(pack);
    for (auto& x : src) {
        x = perf_uniform_i8(seed, 0, 8);
    }
    for (auto& x : weight) {
        x = perf_uniform_i8(seed, 0, 8);
    }
    for (size_t i = 0; i < pack; ++i) {
        bias[i] = perf_uniform_float(seed, -3.0f, 3.0f);
        scale[i] = perf_uniform_float(seed, 0.02f, 0.08f);
    }

    QuanPostTreatParameters parameters = {bias.data(), scale.data(), -127, 127};
    int8_t idxOrder[1] = {0};
    MNNLineDepthWiseInt8AddBiasScaleUnit_scalar(ref.data(), src.data(), weight.data(), &parameters, width, src_w_step,
                                                fw, fh, dilateX_step, dilateY_step, idxOrder);
    MNNLineDepthWiseInt8AddBiasScaleUnit_RVV(out.data(), src.data(), weight.data(), &parameters, width, src_w_step, fw,
                                             fh, dilateX_step, dilateY_step, idxOrder);
    for (size_t i = 0; i < dstSize; ++i) {
        if (std::abs(static_cast<int>(ref[i]) - static_cast<int>(out[i])) > 1) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%d rvv=%d\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNLineDepthWiseInt8AddBiasScaleUnit_scalar : MNNLineDepthWiseInt8AddBiasScaleUnit_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), weight.data(), &parameters, width, src_w_step, fw, fh, dilateX_step, dilateY_step,
           idxOrder);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), weight.data(), &parameters, width, src_w_step, fw, fh, dilateX_step, dilateY_step,
           idxOrder);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

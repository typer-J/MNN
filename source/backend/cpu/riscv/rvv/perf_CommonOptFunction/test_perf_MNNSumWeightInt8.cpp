#include "perf_common.h"

#include "../MNNSumWeightInt8.cpp"

static void MNNSumWeightInt8_scalar(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP) {
    const size_t inside = hP * lP;
    const size_t stride0 = inside * reduceAxis;
    std::vector<float> accum(hP);
    for (size_t i = 0; i < outside; ++i) {
        std::fill(accum.begin(), accum.end(), 0.0f);
        for (size_t j = 0; j < reduceAxis; ++j) {
            for (size_t k = 0; k < hP; ++k) {
                for (size_t x = 0; x < lP; ++x) {
                    accum[k] += static_cast<float>(source[x + k * lP + j * inside + i * stride0]);
                }
            }
        }
        std::memcpy(kernelsum + i * hP, accum.data(), hP * sizeof(float));
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t outside = 64;
    constexpr size_t reduceAxis = 128;
    constexpr size_t hP = 64;
    constexpr size_t lP = 64;
    constexpr int warmup = 20;
    constexpr int loop = 300;
    const size_t srcSize = outside * reduceAxis * hP * lP;
    const size_t dstSize = outside * hP;

    uint64_t seed = 0x5a800001ULL;
    std::vector<int8_t> src(srcSize);
    std::vector<float> ref(dstSize), out(dstSize);
    for (auto& x : src) {
        x = perf_uniform_i8(seed);
    }

    MNNSumWeightInt8_scalar(ref.data(), src.data(), outside, reduceAxis, hP, lP);
    MNNSumWeightInt8(out.data(), src.data(), outside, reduceAxis, hP, lP);
    for (size_t i = 0; i < dstSize; ++i) {
        if (!perf_close(ref[i], out[i], 1e-4f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNSumWeightInt8_scalar : MNNSumWeightInt8;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), outside, reduceAxis, hP, lP);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), outside, reduceAxis, hP, lP);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

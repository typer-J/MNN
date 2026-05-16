#include "perf_common.h"

#include "../MNNMaxPoolInt8.cpp"

static void MNNMaxPoolInt8_scalar(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx,
                                  size_t kernely, size_t stridesx) {
    constexpr size_t pack = 16;
    int8_t* dstPtr = dst;
    const int8_t* srcPtr = src;
    for (size_t ox = 0; ox < outputWidth; ++ox) {
        int8_t result[pack];
        std::fill(result, result + pack, static_cast<int8_t>(INT8_MIN));
        for (size_t y = 0; y < kernely; ++y) {
            for (size_t x = 0; x < kernelx; ++x) {
                const int8_t* inputPtr = srcPtr + pack * (x + inputWidth * y);
                for (size_t idx = 0; idx < pack; ++idx) {
                    result[idx] = std::max(result[idx], inputPtr[idx]);
                }
            }
        }
        std::memcpy(dstPtr, result, pack * sizeof(int8_t));
        dstPtr += pack;
        srcPtr += pack * stridesx;
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t outputWidth = 511;
    constexpr size_t inputWidth = 1024;
    constexpr size_t kernelx = 3;
    constexpr size_t kernely = 3;
    constexpr size_t stridesx = 2;
    constexpr size_t pack = 16;
    constexpr int warmup = 50;
    constexpr int loop = 5000;
    const size_t srcSize = inputWidth * kernely * pack;
    const size_t dstSize = outputWidth * pack;

    uint64_t seed = 0xa5010001ULL;
    std::vector<int8_t> src(srcSize), ref(dstSize), out(dstSize);
    for (auto& x : src) {
        x = perf_uniform_i8(seed);
    }

    MNNMaxPoolInt8_scalar(ref.data(), src.data(), outputWidth, inputWidth, kernelx, kernely, stridesx);
    MNNMaxPoolInt8_RVV(out.data(), src.data(), outputWidth, inputWidth, kernelx, kernely, stridesx);
    for (size_t i = 0; i < dstSize; ++i) {
        if (ref[i] != out[i]) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%d rvv=%d\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNMaxPoolInt8_scalar : MNNMaxPoolInt8_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), outputWidth, inputWidth, kernelx, kernely, stridesx);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), outputWidth, inputWidth, kernelx, kernely, stridesx);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

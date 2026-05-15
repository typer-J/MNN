#include "perf_common.h"

#include "../MNNReorderWeightInt4.cpp"

static void MNNReorderWeightInt4_scalar(uint8_t* dest, const uint8_t* source, int32_t* shape, size_t size, float* kernelsum) {
    MNN_ASSERT(size > 4);
    const int32_t blocknum = shape[0];
    const int32_t hu = shape[1];
    const int32_t lu = shape[2];
    const int32_t hp = shape[3];
    const int32_t lp = shape[4];
    const int32_t ic = blocknum * lu * lp;
    const int32_t stride0 = blocknum * hp * lu * lp;
    const int32_t stride1 = lu * hp * lp;
    const int32_t stride2 = hp * lp;
    for (int32_t i = 0; i < hu; ++i) {
        for (int32_t k = 0; k < hp; ++k) {
            for (int32_t bl = 0; bl < blocknum; ++bl) {
                for (int32_t j = 0; j < lu; ++j) {
                    const int32_t srcIndex = (i * hp + k) * ic + bl * (lu * lp) + j * lp;
                    const int32_t dstIndex = i * stride0 + bl * stride1 + j * stride2 + k * lp;
                    std::memcpy(dest + dstIndex, source + srcIndex, lp);
                }
            }
        }
    }

    const int32_t inside = lp * hp;
    const int32_t outside = blocknum * hu;
    std::vector<uint8_t> buffer(inside);
    for (int32_t i = 0; i < outside; ++i) {
        std::vector<float> accum(hp, 0.0f);
        for (int32_t k = 0; k < lu; ++k) {
            for (int32_t j = 0; j < inside / 2; ++j) {
                const uint8_t w0 = dest[j + (i * lu + k) * inside] >> 4;
                const uint8_t w1 = dest[j + (i * lu + k) * inside] & 0x0f;
                const uint8_t w2 = dest[(i * lu + k) * inside + j + inside / 2] >> 4;
                const uint8_t w3 = dest[(i * lu + k) * inside + j + inside / 2] & 0x0f;
                buffer[2 * j + 0] = static_cast<uint8_t>(w0 * 16 + w2);
                buffer[2 * j + 1] = static_cast<uint8_t>(w1 * 16 + w3);
                accum[j / lp] += static_cast<float>(w0 + w1);
                accum[(j + inside / 2) / lp] += static_cast<float>(w2 + w3);
            }
            std::memcpy(dest + (i * lu + k) * inside, buffer.data(), inside);
        }
        std::memcpy(kernelsum + i * hp, accum.data(), hp * sizeof(float));
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    int32_t shape[5] = {32, 64, 32, 64, 32};
    constexpr int warmup = 10;
    constexpr int loop = 200;
    const size_t srcSize = static_cast<size_t>(shape[1]) * shape[3] * shape[0] * shape[2] * shape[4];
    const size_t sumSize = static_cast<size_t>(shape[0]) * shape[1] * shape[3];

    uint64_t seed = 0x1e040001ULL;
    std::vector<uint8_t> src(srcSize), refDst(srcSize), outDst(srcSize);
    std::vector<float> refSum(sumSize), outSum(sumSize);
    for (auto& x : src) {
        x = perf_uniform_u8(seed);
    }

    MNNReorderWeightInt4_scalar(refDst.data(), src.data(), shape, 5, refSum.data());
    MNNReorderWeightInt4(outDst.data(), src.data(), shape, 5, outSum.data());
    if (std::memcmp(refDst.data(), outDst.data(), srcSize) != 0) {
        std::fprintf(stderr, "verify failed: dest mismatch\n");
        return 1;
    }
    for (size_t i = 0; i < sumSize; ++i) {
        if (!perf_close(refSum[i], outSum[i], 1e-5f)) {
            std::fprintf(stderr, "verify failed at sum %zu: scalar=%f rvv=%f\n", i, refSum[i], outSum[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNReorderWeightInt4_scalar : MNNReorderWeightInt4;
    for (int i = 0; i < warmup; ++i) {
        fn(outDst.data(), src.data(), shape, 5, outSum.data());
    }
    for (int i = 0; i < loop; ++i) {
        fn(outDst.data(), src.data(), shape, 5, outSum.data());
    }
    perf_touch(outDst.data(), outDst.size());
    perf_touch(outSum.data(), outSum.size());
    return 0;
}

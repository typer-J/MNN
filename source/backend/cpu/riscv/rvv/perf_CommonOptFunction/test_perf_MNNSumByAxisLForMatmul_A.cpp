#include "perf_common.h"

#include "../MNNSumByAxisLForMatmul_A.cpp"

static void MNNSumByAxisLForMatmul_A_scalar(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams) {
    int8_t* srcInt8 = source;
    const float* scalePtr = scale;
    const size_t blockNum = sumParams.blockNum;
    const size_t EP = sumParams.DST_XUNIT;
    const size_t LP = sumParams.SRC_UNIT;
    const size_t colBufferUnitSize = sumParams.unitColBufferSize;
    const int oneScale = sumParams.oneScale;
    const size_t LU = sumParams.LU;
    const size_t valid = sumParams.valid;
    const size_t kernelxy = sumParams.kernelxy;
    const size_t blockSizeQuad = LU / blockNum;
    const int inputBlockQuant = sumParams.inputBlock;
    const size_t lastL = valid ? valid : LP;
    const float singleScale = scale[0];

    do {
        const int step = ALIMIN(static_cast<int>(EP), static_cast<int>(realDstCount));
        const int scaleOffset = inputBlockQuant ? (step * blockNum) : step;
        for (size_t k = 0; k < blockNum; ++k) {
            const int8_t* srcX = srcInt8 + k * (step * LP * blockSizeQuad * kernelxy);
            for (int w = 0; w < step; ++w) {
                float dequantScale = singleScale;
                if (oneScale == 0 && inputBlockQuant) {
                    dequantScale = scalePtr[w + k * step];
                } else if (oneScale == 0) {
                    dequantScale = scalePtr[w];
                }
                int64_t sum = 0;
                const int8_t* srcY = srcX + w * LP;
                for (size_t j = 0; j < kernelxy; ++j) {
                    for (size_t i = 0; i < blockSizeQuad; ++i) {
                        const size_t sumSize = i == blockSizeQuad - 1 ? lastL : LP;
                        const int8_t* srcZ = srcY + j * (blockSizeQuad * step * LP) + i * step * LP;
                        for (size_t x = 0; x < sumSize; ++x) {
                            sum += srcZ[x];
                        }
                    }
                }
                dest[w + k * step] = dequantScale * static_cast<float>(sum);
            }
        }
        scalePtr += scaleOffset;
        dest += step * blockNum;
        realDstCount -= step;
        srcInt8 += colBufferUnitSize;
    } while (realDstCount > 0);
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    SumByAxisParams params{8, 128, 128, 128 * 128 * 32 * 8, 0, 256, 128, 9, 1};
    constexpr ssize_t realDstCount = 256;
    constexpr int warmup = 20;
    constexpr int loop = 300;
    const size_t dstSize = params.blockNum * realDstCount;
    const size_t blockSizeQuad = params.LU / params.blockNum;
    const size_t srcSize = realDstCount * params.SRC_UNIT * blockSizeQuad * params.kernelxy * params.blockNum;
    const size_t scaleSize = params.inputBlock ? (realDstCount * params.blockNum) : realDstCount;

    uint64_t seed = 0x51a00001ULL;
    std::vector<int8_t> src(srcSize);
    std::vector<float> scale(scaleSize), ref(dstSize), out(dstSize);
    for (auto& x : src) {
        x = perf_uniform_i8(seed);
    }
    for (auto& x : scale) {
        x = perf_uniform_float(seed, 0.001f, 1.0f);
    }

    MNNSumByAxisLForMatmul_A_scalar(ref.data(), src.data(), scale.data(), realDstCount, params);
    MNNSumByAxisLForMatmul_A(out.data(), src.data(), scale.data(), realDstCount, params);
    for (size_t i = 0; i < dstSize; ++i) {
        if (!perf_close(ref[i], out[i], 20.0f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNSumByAxisLForMatmul_A_scalar : MNNSumByAxisLForMatmul_A;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), scale.data(), realDstCount, params);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), scale.data(), realDstCount, params);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

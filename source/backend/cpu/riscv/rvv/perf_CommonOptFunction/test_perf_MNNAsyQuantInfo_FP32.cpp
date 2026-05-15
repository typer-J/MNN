#include "perf_common.h"

#include "../MNNAsyQuantInfo_FP32.cpp"

static void MNNCountMaxMinValue_scalar(const float* source, float* minVal, float* maxVal, size_t size) {
    float minV = source[0];
    float maxV = source[0];
    for (size_t i = 1; i < size; ++i) {
        minV = std::min(minV, source[i]);
        maxV = std::max(maxV, source[i]);
    }
    *minVal = minV;
    *maxVal = maxV;
}

static void MNNAsyQuantInfo_FP32_scalar(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax, const float* src, const size_t* info) {
    const size_t blockNum = info[0];
    const size_t plane = info[1];
    const size_t innerSide = info[2];
    const size_t DST_XUNIT = info[3];
    const size_t kernelsize = info[5];
    const size_t blockLU = info[6];
    const size_t stride0 = blockNum * blockLU * plane * innerSide;
    const size_t stride1 = blockLU * plane * innerSide;

    if (info[7] == 1) {
        float maxval = 0.0f;
        float minval = 0.0f;
        MNNCountMaxMinValue_scalar(src, &minval, &maxval, kernelsize * stride0);
        if (info[8] == 1 && (maxval - minval) > 1e-7f) {
            if (minval > 0.0f) {
                minval = 0.0f;
            } else if (maxval < 0.0f) {
                maxval = 0.0f;
            }
        }
        const float range = maxval - minval;
        if (range <= 1e-7f) {
            scale[0] = 1.0f;
            qscale[0] = 1.0f;
            qbias[0] = -maxval;
            bias[0] = maxval;
        } else {
            qscale[0] = 255.0f / range;
            scale[0] = range / 255.0f;
            qbias[0] = -minval * 255.0f / range - 128.0f;
            bias[0] = minval + 128.0f * range / 255.0f;
        }
        return;
    }

    for (size_t i = 0; i < plane; ++i) {
        for (size_t bk = 0; bk < blockNum; ++bk) {
            const size_t idx0 = i * innerSide + bk * stride1;
            float maxV = src[idx0];
            float minV = maxV;
            for (size_t n = 0; n < kernelsize; ++n) {
                for (size_t k = 0; k < blockLU; ++k) {
                    for (size_t j = 0; j < innerSide; ++j) {
                        const size_t dataIndex = idx0 + n * stride0 + k * (plane * innerSide) + j;
                        const float value = src[dataIndex];
                        maxV = std::max(maxV, value);
                        minV = std::min(minV, value);
                    }
                }
            }
            const size_t qIndex = i + bk * plane;
            dstMin[qIndex] = minV;
            dstMax[qIndex] = maxV;
        }
    }

    for (size_t i = 0; i < plane; ++i) {
        const size_t step = ALIMIN(DST_XUNIT, plane - (i / DST_XUNIT) * DST_XUNIT);
        const size_t scaleBase = (i / DST_XUNIT) * DST_XUNIT * blockNum + (i % DST_XUNIT);
        for (size_t k = 0; k < blockNum; ++k) {
            const size_t scaleIndex = scaleBase + k * step;
            const size_t qIndex = i + k * plane;
            const float maxV = dstMax[qIndex];
            const float minV = dstMin[qIndex];
            if (std::fabs(maxV - minV) < 1e-7f) {
                qscale[qIndex] = 0.0f;
                qbias[qIndex] = 0.0f;
                scale[scaleIndex] = 0.0f;
                bias[scaleIndex] = maxV;
            } else {
                qscale[qIndex] = 255.0f / (maxV - minV);
                qbias[qIndex] = std::round(-minV * 255.0f / (maxV - minV)) - 128.0f;
                scale[scaleIndex] = (maxV - minV) / 255.0f;
                bias[scaleIndex] = minV + (128.0f / 255.0f) * (maxV - minV);
            }
        }
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    size_t info[9] = {16, 256, 16, 128, 0, 9, 16, 0, 1};
    constexpr int warmup = 20;
    constexpr int loop = 500;
    const size_t blockNum = info[0];
    const size_t plane = info[1];
    const size_t innerSide = info[2];
    const size_t DST_XUNIT = info[3];
    const size_t kernelsize = info[5];
    const size_t blockLU = info[6];
    const size_t srcSize = kernelsize * blockNum * blockLU * plane * innerSide;
    const size_t qSize = blockNum * plane;
    const size_t scaleSize = UP_DIV(plane, DST_XUNIT) * DST_XUNIT * blockNum;

    uint64_t seed = 0xa5a10001ULL;
    std::vector<float> src(srcSize), refScale(scaleSize), refBias(scaleSize), refQScale(qSize), refQBias(qSize), refMin(qSize), refMax(qSize);
    std::vector<float> outScale(scaleSize), outBias(scaleSize), outQScale(qSize), outQBias(qSize), outMin(qSize), outMax(qSize);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -12.0f, 12.0f);
    }

    MNNAsyQuantInfo_FP32_scalar(refScale.data(), refBias.data(), refQScale.data(), refQBias.data(), refMin.data(), refMax.data(), src.data(), info);
    MNNAsyQuantInfo_FP32(outScale.data(), outBias.data(), outQScale.data(), outQBias.data(), outMin.data(), outMax.data(), src.data(), info);
    for (size_t i = 0; i < qSize; ++i) {
        if (!perf_close(refQScale[i], outQScale[i], 1e-4f) || !perf_close(refQBias[i], outQBias[i], 1e-3f) ||
            !perf_close(refMin[i], outMin[i], 1e-6f) || !perf_close(refMax[i], outMax[i], 1e-6f)) {
            std::fprintf(stderr, "verify failed at q %zu\n", i);
            return 1;
        }
    }
    for (size_t i = 0; i < scaleSize; ++i) {
        if (!perf_close(refScale[i], outScale[i], 1e-6f) || !perf_close(refBias[i], outBias[i], 1e-5f)) {
            std::fprintf(stderr, "verify failed at scale %zu\n", i);
            return 1;
        }
    }

    for (int i = 0; i < warmup; ++i) {
        if (mode == 0) {
            MNNAsyQuantInfo_FP32_scalar(outScale.data(), outBias.data(), outQScale.data(), outQBias.data(), outMin.data(), outMax.data(), src.data(), info);
        } else {
            MNNAsyQuantInfo_FP32(outScale.data(), outBias.data(), outQScale.data(), outQBias.data(), outMin.data(), outMax.data(), src.data(), info);
        }
    }
    for (int i = 0; i < loop; ++i) {
        if (mode == 0) {
            MNNAsyQuantInfo_FP32_scalar(outScale.data(), outBias.data(), outQScale.data(), outQBias.data(), outMin.data(), outMax.data(), src.data(), info);
        } else {
            MNNAsyQuantInfo_FP32(outScale.data(), outBias.data(), outQScale.data(), outQBias.data(), outMin.data(), outMax.data(), src.data(), info);
        }
    }
    perf_touch(outQScale.data(), outQScale.size());
    return 0;
}

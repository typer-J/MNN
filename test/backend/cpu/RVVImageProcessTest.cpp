// Copyright © 2026, Alibaba Group Holding Limited
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ImageProcessFunction.hpp"

// The _RVV kernels only exist in the MNNRVV object library, so the direct kernel
// comparisons live under MNN_TEST_RVV_ENABLED. An MNN_USE_RVV=OFF build still
// compiles and links; it just reports the scalar registration instead.
#if MNN_TEST_RVV_ENABLED
void MNNC3ToFloatC3_RVV(const unsigned char*, float*, const float*, const float*, size_t);
void MNNC3ToFloatRGBA_RVV(const unsigned char*, float*, const float*, const float*, size_t);
void MNNSamplerC4Nearest_RVV(const unsigned char*, unsigned char*, MNN::CV::Point*, size_t, size_t, size_t, size_t,
                             size_t, size_t);
void MNNSamplerC4Bilinear_RVV(const unsigned char*, unsigned char*, MNN::CV::Point*, size_t, size_t, size_t, size_t,
                              size_t, size_t);

static bool sameFloat(const std::vector<float>& actual, const std::vector<float>& expected, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(actual.data() + i, expected.data() + i, sizeof(float)) != 0) {
            MNN_ERROR("RVV image float mismatch at %zu: actual=%g expected=%g\n", i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

// Drive the real entry points, not the _RVV symbols. The float blitters are
// reached from ImageProcessUtils::choose(ImageFormat, int) by bare name, so
// calling MNNC3ToFloatC3 here exercises the same code path the pipeline takes.
static bool floatBlitterCase(size_t count) {
    const float mean[3] = {127.0f, -3.25f, 191.5f};
    const float normal[3] = {1.0f / 128.0f, -0.75f, 1.125f};
    std::vector<unsigned char> source(3 * count + 2, 211);
    for (size_t i = 0; i < 3 * count; ++i) {
        source[i + 1] = static_cast<unsigned char>((i * 73 + count * 19 + 11) & 255);
    }
    std::vector<float> c3(3 * count + 4, -999.0f), c3Expected(c3);
    std::vector<float> rgba(4 * count + 4, -999.0f), rgbaExpected(rgba);
    for (size_t i = 0; i < count; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            // Bit-exact oracle: keep the scalar expression order, including the
            // multiply-by-normal-last form the RVV kernel emits.
            const float value = normal[c] * (source[1 + 3 * i + c] - mean[c]);
            c3Expected[1 + 3 * i + c] = value;
            rgbaExpected[1 + 4 * i + c] = value;
        }
        rgbaExpected[1 + 4 * i + 3] = 0.0f;
    }
    MNNC3ToFloatC3(source.data() + 1, c3.data() + 1, mean, normal, count);
    MNNC3ToFloatRGBA(source.data() + 1, rgba.data() + 1, mean, normal, count);
    return sameFloat(c3, c3Expected, c3.size()) && sameFloat(rgba, rgbaExpected, rgba.size());
}

static float clampCoordinate(float value, float maxValue) {
    return std::max(0.0f, std::min(value, maxValue));
}

static void nearestReference(const unsigned char* source, unsigned char* dest, const MNN::CV::Point* points, size_t sta,
                             size_t count, size_t iw, size_t ih, size_t yStride) {
    float x = points[0].fX;
    float y = points[0].fY;
    for (size_t i = 0; i < count; ++i) {
        const int sx = (int)roundf(clampCoordinate(x, (float)(iw - 1)));
        const int sy = (int)roundf(clampCoordinate(y, (float)(ih - 1)));
        std::memcpy(dest + 4 * (sta + i), source + (size_t)sy * yStride + 4 * (size_t)sx, 4);
        x += points[1].fX;
        y += points[1].fY;
    }
}

static void bilinearReference(const unsigned char* source, unsigned char* dest, const MNN::CV::Point* points,
                              size_t sta, size_t count, size_t iw, size_t ih, size_t yStride) {
    float x = points[0].fX;
    float y = points[0].fY;
    for (size_t i = 0; i < count; ++i) {
        const float cx = clampCoordinate(x, (float)(iw - 1));
        const float cy = clampCoordinate(y, (float)(ih - 1));
        const int x0 = (int)cx;
        const int y0 = (int)cy;
        const int x1 = std::min((int)ceilf(cx), (int)iw - 1);
        const int y1 = std::min((int)ceilf(cy), (int)ih - 1);
        const float xF = cx - (float)x0;
        const float yF = cy - (float)y0;
        for (size_t c = 0; c < 4; ++c) {
            const unsigned char c00 = source[(size_t)y0 * yStride + 4 * (size_t)x0 + c];
            const unsigned char c01 = source[(size_t)y0 * yStride + 4 * (size_t)x1 + c];
            const unsigned char c10 = source[(size_t)y1 * yStride + 4 * (size_t)x0 + c];
            const unsigned char c11 = source[(size_t)y1 * yStride + 4 * (size_t)x1 + c];
            // The generic body keeps `1.0 - xF` as a double; this oracle copies
            // that arithmetic verbatim so the comparison is against the original
            // scalar semantics rather than against the new kernel.
            float value = (1.0f - xF) * (1.0f - yF) * c00 + xF * (1.0f - yF) * c01 +
                          yF * (1.0 - xF) * c10 + xF * yF * c11;
            value = std::min(std::max(value, 0.0f), 255.0f);
            dest[4 * (sta + i) + c] = static_cast<unsigned char>(roundf(value));
        }
        x += points[1].fX;
        y += points[1].fY;
    }
}

static bool samplerCase(size_t count, size_t sta, MNN::CV::Point start, MNN::CV::Point delta) {
    const size_t iw = 19, ih = 11, yStride = iw * 4 + 12;
    std::vector<unsigned char> source(yStride * ih, 0);
    for (size_t y = 0; y < ih; ++y) {
        for (size_t x = 0; x < iw; ++x) {
            for (size_t c = 0; c < 4; ++c) {
                source[y * yStride + 4 * x + c] = static_cast<unsigned char>((y * 61 + x * 29 + c * 47) & 255);
            }
        }
    }
    MNN::CV::Point points[2] = {start, delta};
    const size_t outputSize = 4 * (sta + count) + 8;
    std::vector<unsigned char> nearest(outputSize, 0xa5), nearestExpected(nearest);
    std::vector<unsigned char> bilinear(outputSize, 0x5a), bilinearExpected(bilinear);
    nearestReference(source.data(), nearestExpected.data(), points, sta, count, iw, ih, yStride);
    bilinearReference(source.data(), bilinearExpected.data(), points, sta, count, iw, ih, yStride);
    MNNSamplerC4Nearest(source.data(), nearest.data(), points, sta, count, outputSize, iw, ih, yStride);
    MNNSamplerC4Bilinear(source.data(), bilinear.data(), points, sta, count, outputSize, iw, ih, yStride);
    if (nearest != nearestExpected || bilinear != bilinearExpected) {
        MNN_ERROR("RVV image sampler mismatch count=%zu sta=%zu start=(%g,%g) delta=(%g,%g)\n", count, sta,
                  start.fX, start.fY, delta.fX, delta.fY);
        return false;
    }
    return true;
}
#endif // MNN_TEST_RVV_ENABLED

class RVVImageProcessTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto core = MNN::MNNGetCoreFunctions();
        if (!core) {
            MNN_ERROR("RVV image-process test requires an initialized CPU backend\n");
            return false;
        }
#if MNN_TEST_RVV_ENABLED
        // The C4 samplers do go through the function table on the CV::ImageProcess
        // path, so an unregistered slot there is a real bug. The float blitters are
        // picked by bare name and have no live slot, so they are covered by the
        // behavioural cases below instead of by a table assertion.
        const bool samplerRegistered = (core->MNNSamplerC4Nearest == MNNSamplerC4Nearest_RVV) &&
                                       (core->MNNSamplerC4Bilinear == MNNSamplerC4Bilinear_RVV);
        if (core->supportRVV != samplerRegistered) {
            MNN_ERROR("Unexpected RVV image-process sampler registration (supportRVV=%d)\n",
                      static_cast<int>(core->supportRVV));
            return false;
        }
        if (!core->supportRVV) {
            MNN_PRINT("RVV image-process: skipped, runtime reports supportRVV=0\n");
            return true;
        }
        const size_t lengths[] = {0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129};
        size_t cases = 0;
        for (size_t count : lengths) {
            if (!floatBlitterCase(count)) {
                return false;
            }
            if (!samplerCase(count, 3, {-2.25f, 13.0f}, {0.8125f, -0.4375f}) ||
                !samplerCase(count, 1, {18.0f, 10.0f}, {-0.53125f, -0.28125f}) ||
                !samplerCase(count, 2, {0.5f, 0.5f}, {0.5f, 0.25f})) {
                return false;
            }
            cases += 4;
        }
        MNN_PRINT("RVV image-process: %zu cases passed (supportRVV=%d)\n", cases, static_cast<int>(core->supportRVV));
#else
        if (core->MNNSamplerC4Nearest == nullptr || core->MNNSamplerC4Bilinear == nullptr) {
            MNN_ERROR("Unexpected scalar image-process function registration\n");
            return false;
        }
#endif
        return true;
    }
};

MNNTestSuiteRegister(RVVImageProcessTest, "backend/cpu/rvv/image_process");

#else

#include "MNNTestSuite.h"

class RVVImageProcessTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        return true;
    }
};

MNNTestSuiteRegister(RVVImageProcessTest, "backend/cpu/rvv/image_process");

#endif

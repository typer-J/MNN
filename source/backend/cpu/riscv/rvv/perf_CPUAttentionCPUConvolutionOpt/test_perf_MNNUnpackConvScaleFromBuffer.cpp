#include "perf_common.h"

#include "../MNNUnpackConvScaleFromBuffer.cpp"

static void MNNUnpackConvScaleFromBuffer_scalar(float* scaleBuffer, const int8_t* srcbuffer, const int32_t* info,
                                                int infoBytes) {
    const int blockNum = info[0];
    const int ocDiv = info[1];
    const int stride1 = info[2];
    const int unit = info[3];
    const size_t copyBytes = static_cast<size_t>(unit) * infoBytes;
    const size_t packedUnitSize = static_cast<size_t>(stride1) + 2 * copyBytes;
    int8_t* scaleWritePtr = reinterpret_cast<int8_t*>(scaleBuffer);

    for (int hU = 0; hU < ocDiv; ++hU) {
        const int8_t* huPtr = srcbuffer + static_cast<size_t>(hU) * blockNum * packedUnitSize;
        for (int bl = 0; bl < blockNum; ++bl) {
            const int8_t* scaleReadPtr = huPtr + static_cast<size_t>(bl) * packedUnitSize + stride1;
            std::memcpy(scaleWritePtr, scaleReadPtr, copyBytes);
            scaleWritePtr += copyBytes;
        }
    }
}

static bool verify_unpack_case(int blockNum, int ocDiv, int stride1, int unit, int infoBytes, uint64_t& seed) {
    int32_t info[4] = {blockNum, ocDiv, stride1, unit};
    const size_t copyBytes = static_cast<size_t>(unit) * infoBytes;
    const size_t packedUnitSize = static_cast<size_t>(stride1) + 2 * copyBytes;
    const size_t srcSize = static_cast<size_t>(ocDiv) * blockNum * packedUnitSize;
    const size_t dstBytes = static_cast<size_t>(ocDiv) * blockNum * copyBytes;
    const size_t dstFloats = UP_DIV(dstBytes, sizeof(float));
    std::vector<int8_t> src(srcSize);
    std::vector<float> ref(dstFloats), out(dstFloats);
    for (auto& x : src) {
        x = perf_uniform_i8(seed);
    }
    std::fill(ref.begin(), ref.end(), 0.0f);
    std::fill(out.begin(), out.end(), 0.0f);

    MNNUnpackConvScaleFromBuffer_scalar(ref.data(), src.data(), info, infoBytes);
    MNNUnpackConvScaleFromBuffer_RVV(out.data(), src.data(), info, infoBytes);
    if (std::memcmp(ref.data(), out.data(), dstBytes) != 0) {
        std::fprintf(stderr, "verify failed for blockNum=%d ocDiv=%d stride1=%d unit=%d\n", blockNum, ocDiv, stride1,
                     unit);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    uint64_t seed = 0xa77e3001ULL;
    if (!verify_unpack_case(1, 3, 17, 4, 4, seed) || !verify_unpack_case(5, 7, 103, 8, 4, seed) ||
        !verify_unpack_case(3, 11, 257, 16, 2, seed)) {
        return 1;
    }

    constexpr int blockNum = 16;
    constexpr int ocDiv = 256;
    constexpr int stride1 = 4096;
    constexpr int unit = 16;
    constexpr int infoBytes = 4;
    constexpr int warmup = 40;
    constexpr int loop = 1000;
    int32_t info[4] = {blockNum, ocDiv, stride1, unit};
    const size_t copyBytes = static_cast<size_t>(unit) * infoBytes;
    const size_t packedUnitSize = static_cast<size_t>(stride1) + 2 * copyBytes;
    const size_t srcSize = static_cast<size_t>(ocDiv) * blockNum * packedUnitSize;
    const size_t dstBytes = static_cast<size_t>(ocDiv) * blockNum * copyBytes;
    const size_t dstFloats = UP_DIV(dstBytes, sizeof(float));
    std::vector<int8_t> src(srcSize);
    std::vector<float> dst(dstFloats);
    for (auto& x : src) {
        x = perf_uniform_i8(seed);
    }

    auto fn = mode == 0 ? MNNUnpackConvScaleFromBuffer_scalar : MNNUnpackConvScaleFromBuffer_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(dst.data(), src.data(), info, infoBytes);
    }
    for (int i = 0; i < loop; ++i) {
        fn(dst.data(), src.data(), info, infoBytes);
    }
    perf_touch(dst.data(), dst.size());
    return 0;
}

#include "perf_common.h"

#include "../MNNAttenPackAndScaleSingleHead.cpp"

static void MNNAttenPackAndScaleSingleHead_scalar(float* dst, const float* srcHeadBase, size_t srcRowStride,
                                                  const float* scale, const int32_t* units, size_t seqLen,
                                                  size_t headDim) {
    const int32_t eP = units[0];
    const int32_t lP = units[1];
    const float scaleVal = scale[0];
    const size_t packedHeadDim = UP_DIV(headDim, lP);
    const size_t dstStrideDOuter = static_cast<size_t>(eP) * lP;
    const size_t dstStrideSOuter = packedHeadDim * dstStrideDOuter;

    for (size_t s = 0; s < seqLen; ++s) {
        const size_t sOuter = s / eP;
        const size_t sInner = s % eP;
        const float* srcRowPtr = srcHeadBase + s * srcRowStride;
        float* dstBasePtr = dst + sOuter * dstStrideSOuter + sInner * lP;
        for (size_t d = 0; d < headDim; ++d) {
            dstBasePtr[d * dstStrideDOuter] = srcRowPtr[d] * scaleVal;
        }
    }
}

static bool verify_pack_case(size_t seqLen, size_t headDim, int eP, int lP, uint64_t& seed) {
    const size_t srcRowStride = headDim + 17;
    const size_t packedHeadDim = UP_DIV(headDim, lP);
    const size_t dstSize = UP_DIV(seqLen, eP) * packedHeadDim * eP * lP;
    const size_t srcSize = (seqLen - 1) * srcRowStride + headDim;
    std::vector<float> src(srcSize), ref(dstSize), out(dstSize);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -3.0f, 3.0f);
    }
    std::fill(ref.begin(), ref.end(), -1.0f);
    std::fill(out.begin(), out.end(), -1.0f);
    const float scale = 0.375f;
    const int32_t units[2] = {eP, lP};
    MNNAttenPackAndScaleSingleHead_scalar(ref.data(), src.data(), srcRowStride, &scale, units, seqLen, headDim);
    MNNAttenPackAndScaleSingleHead_RVV(out.data(), src.data(), srcRowStride, &scale, units, seqLen, headDim);
    for (size_t i = 0; i < dstSize; ++i) {
        if (!perf_close(ref[i], out[i], 1e-6f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    uint64_t seed = 0xa77e1001ULL;
    if (!verify_pack_case(3, 7, 4, 1, seed) || !verify_pack_case(13, 97, 8, 1, seed) ||
        !verify_pack_case(17, 255, 12, 1, seed)) {
        return 1;
    }

    constexpr size_t seqLen = 128;
    constexpr size_t headDim = 4096;
    constexpr int eP = 12;
    constexpr int lP = 1;
    constexpr int warmup = 50;
    constexpr int loop = 1200;
    const size_t srcRowStride = headDim * 4;
    const size_t dstSize = UP_DIV(seqLen, eP) * UP_DIV(headDim, lP) * eP * lP;
    const size_t srcSize = (seqLen - 1) * srcRowStride + headDim;
    std::vector<float> src(srcSize), dst(dstSize);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -3.0f, 3.0f);
    }
    const float scale = 0.08838835f;
    const int32_t units[2] = {eP, lP};
    auto fn = mode == 0 ? MNNAttenPackAndScaleSingleHead_scalar : MNNAttenPackAndScaleSingleHead_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(dst.data(), src.data(), srcRowStride, &scale, units, seqLen, headDim);
    }
    for (int i = 0; i < loop; ++i) {
        fn(dst.data(), src.data(), srcRowStride, &scale, units, seqLen, headDim);
    }
    perf_touch(dst.data(), dst.size());
    return 0;
}

#include "perf_common.h"

#include "../MNNAttentionMaskQK.cpp"

static void MNNAttentionMaskQK_scalar(float* qkPacked, const float* scale, size_t seqLen, size_t processedKvSeq,
                                      int pack, int kvSeqLen, int kvoffset, int padKvSeqLen, const float* sinksPtr,
                                      const float* maskPtr, size_t maskElementSize, bool quantKey,
                                      bool isLowerTriangular) {
    (void)sinksPtr;
    if (isLowerTriangular && quantKey) {
        return;
    }
    const float scaleVal = scale[0];
    const int gapLen = (maskElementSize == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen))
                           ? 0
                           : static_cast<int>(kvSeqLen - seqLen);
    const size_t kvBlockCount = UP_DIV(processedKvSeq, pack);
    const size_t qkSize = ROUND_UP(processedKvSeq, pack) * seqLen;

    if (isLowerTriangular) {
        for (size_t i = 0; i < qkSize; ++i) {
            qkPacked[i] *= scaleVal;
        }
        return;
    }
    if (maskPtr == nullptr) {
        return;
    }

    const int maskCols = (maskElementSize == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen))
                             ? kvSeqLen + padKvSeqLen
                             : static_cast<int>(seqLen) + padKvSeqLen;
    for (size_t i = 0; i < kvBlockCount; ++i) {
        float* blockDataPtr = qkPacked + i * seqLen * pack;
        for (size_t j = 0; j < seqLen; ++j) {
            float* dataPtr = blockDataPtr + j * pack;
            const float* currentMaskRow = maskPtr + j * maskCols;
            for (int k = 0; k < pack; ++k) {
                float val = dataPtr[k];
                if (!quantKey) {
                    val *= scaleVal;
                    dataPtr[k] = val;
                }
                const int currentKvSeqIndex = kvoffset + static_cast<int>(i) * pack + k;
                if (currentKvSeqIndex < gapLen) {
                    continue;
                }
                if (currentKvSeqIndex - gapLen >= maskCols) {
                    break;
                }
                val += currentMaskRow[currentKvSeqIndex - gapLen];
                dataPtr[k] = val;
            }
        }
    }
}

static bool verify_mask_case(size_t seqLen, size_t processedKvSeq, int kvSeqLen, int kvoffset, int padKvSeqLen,
                             bool quantKey, bool isLowerTriangular, bool fullMask, uint64_t& seed) {
    constexpr int pack = 4;
    const size_t maskCols = fullMask ? kvSeqLen + padKvSeqLen : seqLen + padKvSeqLen;
    const size_t maskElementSize =
        fullMask ? (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen) : (seqLen + padKvSeqLen) * (seqLen + padKvSeqLen);
    const size_t qkSize = ROUND_UP(processedKvSeq, pack) * seqLen;
    std::vector<float> mask(maskElementSize);
    std::vector<float> ref(qkSize), out(qkSize);
    for (auto& x : ref) {
        x = perf_uniform_float(seed, -4.0f, 4.0f);
    }
    out = ref;
    for (size_t i = 0; i < maskElementSize; ++i) {
        mask[i] = (i % maskCols) > (i / maskCols) ? -10000.0f : perf_uniform_float(seed, -0.5f, 0.5f);
    }
    const float scale = 0.125f;
    const float* maskPtr = isLowerTriangular ? nullptr : mask.data();
    const size_t actualMaskSize = isLowerTriangular ? 0 : maskElementSize;

    MNNAttentionMaskQK_scalar(ref.data(), &scale, seqLen, processedKvSeq, pack, kvSeqLen, kvoffset, padKvSeqLen,
                              nullptr, maskPtr, actualMaskSize, quantKey, isLowerTriangular);
    MNNAttentionMaskQK_RVV(out.data(), &scale, seqLen, processedKvSeq, pack, kvSeqLen, kvoffset, padKvSeqLen, nullptr,
                           maskPtr, actualMaskSize, quantKey, isLowerTriangular);
    for (size_t i = 0; i < qkSize; ++i) {
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

    uint64_t seed = 0xa77e0001ULL;
    if (!verify_mask_case(5, 9, 23, 16, 0, false, false, false, seed) ||
        !verify_mask_case(7, 13, 64, 8, 3, true, false, true, seed) ||
        !verify_mask_case(11, 21, 21, 0, 0, false, true, false, seed)) {
        return 1;
    }

    constexpr size_t seqLen = 64;
    constexpr size_t processedKvSeq = 1024;
    constexpr int pack = 4;
    constexpr int kvSeqLen = 1024;
    constexpr int kvoffset = 0;
    constexpr int padKvSeqLen = 0;
    constexpr int warmup = 50;
    constexpr int loop = 1000;
    const size_t qkSize = ROUND_UP(processedKvSeq, pack) * seqLen;
    const size_t maskElementSize = seqLen * kvSeqLen;
    std::vector<float> qk(qkSize), mask(maskElementSize);
    for (auto& x : qk) {
        x = perf_uniform_float(seed, -6.0f, 6.0f);
    }
    for (size_t i = 0; i < maskElementSize; ++i) {
        mask[i] = (i % kvSeqLen) > (i / kvSeqLen + 64) ? -10000.0f : perf_uniform_float(seed, -0.25f, 0.25f);
    }
    const float scale = 0.08838835f;
    auto fn = mode == 0 ? MNNAttentionMaskQK_scalar : MNNAttentionMaskQK_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(qk.data(), &scale, seqLen, processedKvSeq, pack, kvSeqLen, kvoffset, padKvSeqLen, nullptr, mask.data(),
           maskElementSize, false, false);
    }
    for (int i = 0; i < loop; ++i) {
        fn(qk.data(), &scale, seqLen, processedKvSeq, pack, kvSeqLen, kvoffset, padKvSeqLen, nullptr, mask.data(),
           maskElementSize, false, false);
    }
    perf_touch(qk.data(), qk.size());
    return 0;
}

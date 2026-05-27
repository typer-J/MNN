#include "perf_common.h"

#include "../MNNFlashAttentionUpdateBlockOutput.cpp"

static void MNNFlashAttentionUpdateBlockOutput_scalar(float* dst, float* src, float* scale, float* normalizeScale,
                                                      int depthQuad, int plane, int pack, int idx, int kvBlocks,
                                                      int size, int bytes, int seqStart) {
    const int stride0 = plane * pack;
    if (idx > 0) {
        for (int j = 0; j < depthQuad; ++j) {
            for (int i = seqStart; i < plane; ++i) {
                for (int k = 0; k < pack; ++k) {
                    const int offset = j * stride0 + i * pack + k;
                    dst[offset] = src[offset] + dst[offset] * scale[i];
                }
            }
        }
    } else {
        std::memcpy(dst, src, static_cast<size_t>(size) * bytes);
    }
    if (idx == kvBlocks - 1) {
        for (int j = 0; j < depthQuad; ++j) {
            for (int i = 0; i < plane; ++i) {
                const float normalize = 1.0f / normalizeScale[i];
                for (int k = 0; k < pack; ++k) {
                    dst[j * stride0 + i * pack + k] *= normalize;
                }
            }
        }
    }
}

static bool verify_update_case(int depthQuad, int plane, int pack, int idx, int kvBlocks, int seqStart,
                               uint64_t& seed) {
    const int size = depthQuad * plane * pack;
    std::vector<float> src(size), ref(size), out(size), scale(plane), normalize(plane);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -2.0f, 2.0f);
    }
    for (int i = 0; i < size; ++i) {
        ref[i] = out[i] = perf_uniform_float(seed, -1.0f, 1.0f);
    }
    for (int i = 0; i < plane; ++i) {
        scale[i] = perf_uniform_float(seed, 0.1f, 0.9f);
        normalize[i] = perf_uniform_float(seed, 0.5f, 2.5f);
    }

    MNNFlashAttentionUpdateBlockOutput_scalar(ref.data(), src.data(), scale.data(), normalize.data(), depthQuad, plane,
                                              pack, idx, kvBlocks, size, sizeof(float), seqStart);
    MNNFlashAttentionUpdateBlockOutput_RVV(out.data(), src.data(), scale.data(), normalize.data(), depthQuad, plane,
                                           pack, idx, kvBlocks, size, sizeof(float), seqStart);
    for (int i = 0; i < size; ++i) {
        if (!perf_close(ref[i], out[i], 1e-6f)) {
            std::fprintf(stderr, "verify failed at %d: scalar=%f rvv=%f\n", i, ref[i], out[i]);
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

    uint64_t seed = 0xa77e2001ULL;
    if (!verify_update_case(3, 5, 4, 0, 1, 0, seed) || !verify_update_case(5, 9, 4, 1, 3, 2, seed) ||
        !verify_update_case(7, 11, 4, 2, 3, 0, seed)) {
        return 1;
    }

    constexpr int depthQuad = 512;
    constexpr int plane = 128;
    constexpr int pack = 4;
    constexpr int kvBlocks = 4;
    constexpr int idx = 3;
    constexpr int seqStart = 7;
    constexpr int warmup = 40;
    constexpr int loop = 1000;
    const int size = depthQuad * plane * pack;
    std::vector<float> src(size), dst(size), scale(plane), normalize(plane);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -2.0f, 2.0f);
    }
    for (auto& x : dst) {
        x = perf_uniform_float(seed, -1.0f, 1.0f);
    }
    for (int i = 0; i < plane; ++i) {
        scale[i] = perf_uniform_float(seed, 0.1f, 0.9f);
        normalize[i] = perf_uniform_float(seed, 0.5f, 2.5f);
    }

    auto fn = mode == 0 ? MNNFlashAttentionUpdateBlockOutput_scalar : MNNFlashAttentionUpdateBlockOutput_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(dst.data(), src.data(), scale.data(), normalize.data(), depthQuad, plane, pack, idx, kvBlocks, size,
           sizeof(float), seqStart);
    }
    for (int i = 0; i < loop; ++i) {
        fn(dst.data(), src.data(), scale.data(), normalize.data(), depthQuad, plane, pack, idx, kvBlocks, size,
           sizeof(float), seqStart);
    }
    perf_touch(dst.data(), dst.size());
    return 0;
}

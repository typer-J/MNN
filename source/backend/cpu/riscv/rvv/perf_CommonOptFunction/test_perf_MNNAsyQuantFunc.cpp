#include "perf_common.h"

#include "../MNNAsyQuantFunc.cpp"

static void MNNAsyQuantFunc_scalar(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
    const size_t blockNum = info[0];
    const size_t EP = info[1];
    const size_t LP = info[2];
    const size_t kernelsize = info[5];
    const size_t blockLU = info[6];
    const size_t stride0 = blockNum * blockLU * EP * LP;
    const size_t stride1 = blockLU * EP * LP;

    for (size_t i = 0; i < EP; ++i) {
        for (size_t bk = 0; bk < blockNum; ++bk) {
            const float quantScale = qscale[i + bk * EP];
            const float quantBias = qbias[i + bk * EP];
            for (size_t n = 0; n < kernelsize; ++n) {
                for (size_t k = 0; k < blockLU; ++k) {
                    for (size_t j = 0; j < LP; ++j) {
                        const size_t idx = n * stride0 + bk * stride1 + k * EP * LP + i * LP + j;
                        int qval = static_cast<int>(std::round(src[idx] * quantScale + quantBias));
                        qval = std::max(-128, std::min(127, qval));
                        dst[idx] = static_cast<int8_t>(qval);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    size_t info[7] = {16, 64, 32, 0, 0, 9, 16};
    constexpr int warmup = 20;
    constexpr int loop = 500;
    const size_t total = info[5] * info[0] * info[6] * info[1] * info[2];
    const size_t scaleSize = info[0] * info[1];

    uint64_t seed = 0xa5f00001ULL;
    std::vector<float> src(total), qscale(scaleSize), qbias(scaleSize);
    std::vector<int8_t> ref(total), out(total);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -10.0f, 10.0f);
    }
    for (size_t i = 0; i < scaleSize; ++i) {
        qscale[i] = perf_uniform_float(seed, 0.5f, 2.5f);
        qbias[i] = perf_uniform_float(seed, -8.0f, 8.0f);
    }

    MNNAsyQuantFunc_scalar(ref.data(), src.data(), qscale.data(), qbias.data(), info);
    MNNAsyQuantFunc(out.data(), src.data(), qscale.data(), qbias.data(), info);
    for (size_t i = 0; i < total; ++i) {
        if (std::abs(static_cast<int>(ref[i]) - static_cast<int>(out[i])) > 1) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%d rvv=%d\n", i, ref[i], out[i]);
            return 1;
        }
    }

    auto fn = mode == 0 ? MNNAsyQuantFunc_scalar : MNNAsyQuantFunc;
    for (int i = 0; i < warmup; ++i) {
        fn(out.data(), src.data(), qscale.data(), qbias.data(), info);
    }
    for (int i = 0; i < loop; ++i) {
        fn(out.data(), src.data(), qscale.data(), qbias.data(), info);
    }
    perf_touch(out.data(), out.size());
    return 0;
}

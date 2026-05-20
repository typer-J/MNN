#include "perf_common.h"

#include "../MNNAccumulateSequenceNumber.cpp"

static void MNNAccumulateSequenceNumber_scalar(float* dst, const float* src, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += src[i];
    }
    *dst = sum;
}

static bool verify_accumulate_case(int size, uint64_t& seed) {
    std::vector<float> src(size > 0 ? size : 1);
    for (int i = 0; i < size; ++i) {
        src[i] = perf_uniform_float(seed, -10.0f, 10.0f);
    }
    float ref = 0.0f;
    float out = 0.0f;
    MNNAccumulateSequenceNumber_scalar(&ref, src.data(), size);
    MNNAccumulateSequenceNumber(&out, src.data(), size);
    const float eps = 1e-2f * std::max(size, 1) / 1024.0f + 1e-5f;
    if (!perf_close(ref, out, eps)) {
        std::fprintf(stderr, "verify failed for size=%d: scalar=%f rvv=%f\n", size, ref, out);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr int size = 1 << 20;
    constexpr int warmup = 100;
    constexpr int loop = 4000;
    uint64_t seed = 0xacc50001ULL;
    std::vector<float> src(size);
    for (auto& x : src) {
        x = perf_uniform_float(seed, -10.0f, 10.0f);
    }

    if (!verify_accumulate_case(0, seed) || !verify_accumulate_case(3, seed) ||
        !verify_accumulate_case(1009, seed)) {
        return 1;
    }

    float ref = 0.0f;
    float out = 0.0f;
    MNNAccumulateSequenceNumber_scalar(&ref, src.data(), size);
    MNNAccumulateSequenceNumber(&out, src.data(), size);
    if (!perf_close(ref, out, 1e-2f * size / 1024.0f)) {
        std::fprintf(stderr, "verify failed: scalar=%f rvv=%f\n", ref, out);
        return 1;
    }

    auto fn = mode == 0 ? MNNAccumulateSequenceNumber_scalar : MNNAccumulateSequenceNumber;
    for (int i = 0; i < warmup; ++i) {
        fn(&out, src.data(), size);
    }
    for (int i = 0; i < loop; ++i) {
        fn(&out, src.data(), size);
    }
    perf_keep(out);
    return 0;
}

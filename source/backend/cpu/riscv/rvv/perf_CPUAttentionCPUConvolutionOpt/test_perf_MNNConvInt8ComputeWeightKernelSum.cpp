#include "perf_common.h"

#include "../MNNConvInt8ComputeWeightKernelSum.cpp"

static void MNNConvInt8ComputeWeightKernelSum_scalar(int* kernelSum, int32_t* bias, const int8_t* weight, int kernelNum,
                                                     int kernelSize, const float* scale, const float* weightBias,
                                                     bool compensateSseOffset) {
    for (int i = 0; i < kernelNum; ++i) {
        int temp = 0;
        const int offset = i * kernelSize;
        for (int j = 0; j < kernelSize; ++j) {
            temp += static_cast<int>(weight[offset + j]);
        }
        kernelSum[i] = static_cast<int>(temp + kernelSize * (weightBias[i] / scale[i]));
        if (compensateSseOffset) {
            bias[i] -= 128 * temp;
        }
    }
}

static bool verify_kernel_sum_case(int kernelNum, int kernelSize, bool compensateSseOffset, uint64_t& seed) {
    const size_t weightSize = static_cast<size_t>(kernelNum) * kernelSize;
    std::vector<int8_t> weight(weightSize);
    std::vector<float> scale(kernelNum), weightBias(kernelNum);
    std::vector<int32_t> refBias(kernelNum), outBias(kernelNum);
    std::vector<int> ref(kernelNum), out(kernelNum);
    for (auto& x : weight) {
        x = perf_uniform_i8(seed);
    }
    for (int i = 0; i < kernelNum; ++i) {
        scale[i] = perf_uniform_float(seed, 0.005f, 0.05f);
        weightBias[i] = perf_uniform_float(seed, -0.25f, 0.25f);
        refBias[i] = outBias[i] = perf_uniform_int(seed, -100000, 100000);
    }
    MNNConvInt8ComputeWeightKernelSum_scalar(ref.data(), refBias.data(), weight.data(), kernelNum, kernelSize,
                                             scale.data(), weightBias.data(), compensateSseOffset);
    MNNConvInt8ComputeWeightKernelSum_RVV(out.data(), outBias.data(), weight.data(), kernelNum, kernelSize,
                                          scale.data(), weightBias.data(), compensateSseOffset);
    for (int i = 0; i < kernelNum; ++i) {
        if (!perf_close_int(ref[i], out[i]) || !perf_close_int(refBias[i], outBias[i])) {
            std::fprintf(stderr, "verify failed at %d: sum scalar=%d rvv=%d bias scalar=%d rvv=%d\n", i, ref[i], out[i],
                         refBias[i], outBias[i]);
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

    uint64_t seed = 0xa77e5001ULL;
    if (!verify_kernel_sum_case(3, 7, false, seed) || !verify_kernel_sum_case(17, 31, true, seed) ||
        !verify_kernel_sum_case(65, 257, false, seed)) {
        return 1;
    }

    constexpr int kernelNum = 4096;
    constexpr int kernelSize = 1024;
    constexpr bool compensateSseOffset = false;
    constexpr int warmup = 20;
    constexpr int loop = 300;
    const size_t weightSize = static_cast<size_t>(kernelNum) * kernelSize;
    std::vector<int8_t> weight(weightSize);
    std::vector<float> scale(kernelNum), weightBias(kernelNum);
    std::vector<int32_t> bias(kernelNum);
    std::vector<int> kernelSum(kernelNum);
    for (auto& x : weight) {
        x = perf_uniform_i8(seed);
    }
    for (int i = 0; i < kernelNum; ++i) {
        scale[i] = perf_uniform_float(seed, 0.005f, 0.05f);
        weightBias[i] = perf_uniform_float(seed, -0.25f, 0.25f);
        bias[i] = perf_uniform_int(seed, -100000, 100000);
    }

    auto fn = mode == 0 ? MNNConvInt8ComputeWeightKernelSum_scalar : MNNConvInt8ComputeWeightKernelSum_RVV;
    for (int i = 0; i < warmup; ++i) {
        fn(kernelSum.data(), bias.data(), weight.data(), kernelNum, kernelSize, scale.data(), weightBias.data(),
           compensateSseOffset);
    }
    for (int i = 0; i < loop; ++i) {
        fn(kernelSum.data(), bias.data(), weight.data(), kernelNum, kernelSize, scale.data(), weightBias.data(),
           compensateSseOffset);
    }
    perf_touch(kernelSum.data(), kernelSum.size());
    return 0;
}

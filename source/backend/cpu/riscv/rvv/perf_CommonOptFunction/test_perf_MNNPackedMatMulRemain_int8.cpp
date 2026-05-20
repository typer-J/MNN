#define PERF_MATMUL_HELPERS_ONLY
#include "test_perf_MNNPackedMatMul_int8.cpp"

int main(int argc, char** argv) {
    const int mode = perf_mode(argc, argv);
    if (mode < 0) {
        return 2;
    }

    constexpr size_t L = 256;
    constexpr size_t H = 128;
    constexpr size_t eSize = 7;
    constexpr int warmup = 50;
    constexpr int loop = 6000;
    size_t parameter[7] = {16 * sizeof(float), L, H, H * 4 * sizeof(float), 0, 0, 0};
    float postParams[4] = {0.0f, 0.0f, -10000.0f, 10000.0f};
    const size_t hC4 = UP_DIV(H, 4);
    const size_t ASize = 16 * L;
    const size_t BSize = hC4 * (4 * L + parameter[5]);
    const size_t CSize = hC4 * (parameter[3] / sizeof(float));

    uint64_t seed = 0x9a7e0001ULL;
    std::vector<float> A(ASize), ref(CSize), out(CSize), bias(hC4 * 4), k(hC4 * 4), b(hC4 * 4);
    std::vector<int8_t> B(BSize);
    for (auto& x : A) x = perf_uniform_float(seed, -1.0f, 1.0f);
    for (auto& x : B) x = perf_uniform_i8(seed);
    for (auto& x : bias) x = perf_uniform_float(seed, -1.0f, 1.0f);
    for (auto& x : k) x = perf_uniform_float(seed, 0.001f, 0.05f);
    for (auto& x : b) x = perf_uniform_float(seed, -0.2f, 0.2f);

    std::fill(ref.begin(), ref.end(), 0.0f);
    std::fill(out.begin(), out.end(), 0.0f);
    MNNPackedMatMulRemain_int8_scalar_impl(ref.data(), A.data(), reinterpret_cast<const float*>(B.data()), eSize, parameter, postParams, bias.data(), 16, k.data(), b.data());
    MNNPackedMatMulRemain_int8(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), eSize, parameter, postParams, bias.data(), k.data(), b.data());
    for (size_t i = 0; i < CSize; ++i) {
        if (!perf_close(ref[i], out[i], 1e-3f)) {
            std::fprintf(stderr, "verify failed at %zu: scalar=%f rvv=%f\n", i, ref[i], out[i]);
            return 1;
        }
    }
    if (!verify_MNNPackedMatMul_int8_case("remain tail blockId", 13, 16, 1, true)) {
        return 1;
    }

    for (int i = 0; i < warmup; ++i) {
        if (mode == 0) {
            MNNPackedMatMulRemain_int8_scalar_impl(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), eSize, parameter, postParams, bias.data(), 16, k.data(), b.data());
        } else {
            MNNPackedMatMulRemain_int8(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), eSize, parameter, postParams, bias.data(), k.data(), b.data());
        }
    }
    for (int i = 0; i < loop; ++i) {
        if (mode == 0) {
            MNNPackedMatMulRemain_int8_scalar_impl(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), eSize, parameter, postParams, bias.data(), 16, k.data(), b.data());
        } else {
            MNNPackedMatMulRemain_int8(out.data(), A.data(), reinterpret_cast<const float*>(B.data()), eSize, parameter, postParams, bias.data(), k.data(), b.data());
        }
    }
    perf_touch(out.data(), out.size());
    return 0;
}

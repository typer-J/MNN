#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include "baseline.hpp"
extern "C" void panel(const float*, const float*, float*, const float*, int, int, int, int);
extern "C" void direct(const float*, const float*, float*, const float*, int, int, int);
extern "C" void post(float*, const float*, const float*, const float*, int, int, int);
#ifndef EMULATE
void MNNPackedMatMulRemainFP32_RVV(float*, const float*, const float*, size_t, const size_t*, const float*,
                                   const float*, const float*, const float*);
#endif
static volatile float sink = 0;
static int up(int n, int p) {
    return (n + p - 1) / p * p;
}
static void fail(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}
static void pack(const float* b, float* w, int k, int h, int p, bool trans) {
    for (int hb = 0; hb < up(h, p) / p; ++hb)
        for (int z = 0; z < k; ++z)
            for (int lane = 0; lane < p; ++lane) {
                int y = hb * p + lane;
                w[(hb * k + z) * p + lane] = y < h ? b[trans ? y * k + z : z * h + y] : 0;
            }
}
static void convert(const float* src, float* dst, int e, int h, int from, int to) {
    for (int y = 0; y < up(h, to); ++y)
        for (int row = 0; row < e; ++row)
            dst[(y / to * e + row) * to + y % to] = y < h ? src[(y / from * e + row) * from + y % from] : 0;
}
struct Guard {
    std::vector<float> v;
    size_t n;
    explicit Guard(size_t size) : v(size + 32, 1234567.0f), n(size) {}
    float* data() { return v.data() + 16; }
    void check() const {
        for (int i = 0; i < 16; ++i)
            if (v[i] != 1234567.0f || v[n + 16 + i] != 1234567.0f)
                fail("guard damaged");
    }
};
static double timeCall(const std::function<void()>& fn, size_t n) {
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i)
        fn();
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / n;
}
int main(int argc, char** argv) {
    bool correctness = argc > 1;
    std::mt19937 rng(20260910);
    std::uniform_real_distribution<float> random(-0.25f, 0.25f);
    struct Shape {
        int e, k, h;
    };
    const Shape shapes[] = {{1, 0, 0},      {1, 0, 3},       {1, 1, 1},     {1, 3, 3},       {1, 7, 7},
                            {1, 15, 15},    {1, 16, 16},     {1, 17, 17},   {1, 31, 33},     {1, 63, 65},
                            {1, 128, 128},  {1, 256, 256},   {1, 512, 512}, {1, 1024, 1024}, {1, 896, 4864},
                            {1, 4864, 896}, {1, 2048, 2048}, {3, 7, 3},     {3, 31, 17},     {8, 64, 64},
                            {16, 128, 128}, {33, 127, 129},  {64, 256, 256}};
    std::printf("e,k,h,transpose,bias,variant,phase,round,iterations,ns\n");
    int checked = 0;
    for (auto s : shapes)
        for (int trans = 0; trans < 2; ++trans)
            for (int biasOn = 0; biasOn < 2; ++biasOn) {
                int e = s.e, k = s.k, h = s.h;
                std::vector<float> a(std::max(1, e * k)), b(std::max(1, k * h)), bias(up(h, 16) + 16, 0);
                std::vector<float> scale(up(h, 16) + 16, 0), shift(up(h, 16) + 16, 0), slope(up(h, 16) + 16, 0);
                for (auto& x : a)
                    x = random(rng);
                for (auto& x : b)
                    x = random(rng);
                for (int y = 0; y < h; ++y) {
                    bias[y] = random(rng);
                    scale[y] = random(rng) * 3;
                    shift[y] = random(rng);
                    slope[y] = random(rng);
                }
                auto aOriginal = a, bOriginal = b;
                std::vector<double> oracle(e * h);
                for (int row = 0; row < e; ++row)
                    for (int y = 0; y < h; ++y) {
                        double x = biasOn ? bias[y] : 0;
                        for (int z = 0; z < k; ++z)
                            x += double(a[row * k + z]) * b[trans ? y * k + z : z * h + y];
                        oracle[row * h + y] = x;
                    }
                auto check = [&](float* result, int p, bool chained) {
                    for (int row = 0; row < e; ++row)
                        for (int y = 0; y < h; ++y) {
                            double ref = oracle[row * h + y];
                            if (chained) {
                                ref = ref * scale[y] + shift[y];
                                if (ref < 0)
                                    ref *= slope[y];
                            }
                            float actual = result[(y / p * e + row) * p + y % p];
                            if (!std::isfinite(actual) || std::abs(actual - ref) > 2e-5 * (1 + std::abs(ref)))
                                fail("oracle mismatch");
                        }
                    ++checked;
                };
                Guard c4(e * up(h, 4)), raw(e * h), ac4(e * up(k, 4));
                // External activation/output ABI is C4. Raw row-major = one full-width panel.
                if (k)
                    convert(a.data(), ac4.data(), e, k, k, 4);
                struct Job {
                    const char* variant;
                    const char* phase;
                    std::function<void()> fn;
                    size_t iterations;
                };
                std::vector<Job> jobs;
                MatMulParam param = {1, k, h, 1, false, bool(trans)};
                auto scalar = [&]() {
                    for (int row = 0; row < e; ++row)
                        MNNComputeMatMulForE_1(a.data() + row * k, b.data(), raw.data() + row * h,
                                               biasOn ? bias.data() : nullptr, &param, 0);
                };
                scalar();
                if (h)
                    check(raw.data(), h, false);
                raw.check();
                // E=1 is the exact original API; E>1 here repeats that API as an exploratory baseline, not production
                // GEMM.
                jobs.push_back({"original_e1", "compute", scalar, 1});
                if (!trans) {
                    auto fn = [&]() {
                        direct(a.data(), b.data(), raw.data(), biasOn ? bias.data() : nullptr, e, k, h);
                    };
                    fn();
                    if (h)
                        check(raw.data(), h, false);
                    raw.check();
                    jobs.push_back({"direct_rvv", "compute", fn, 1});
                }
                Guard weights4(k * up(h, 4)), weights8(k * up(h, 8)), weights16(k * up(h, 16));
                Guard out4(e * up(h, 4)), out8(e * up(h, 8)), out16(e * up(h, 16));
                Guard* weights[] = {&weights4, &weights8, &weights16};
                Guard* outputs[] = {&out4, &out8, &out16};
                const char* names[] = {"panel4", "panel8", "panel16"};
#ifndef EMULATE
                if (e == 1) {
                    pack(b.data(), weights4.data(), k, h, 4, trans);
                    auto old = [&]() {
                        size_t parameter[] = {4, size_t(k), size_t(h), 16, 0, 0};
                        MNNPackedMatMulRemainFP32_RVV(out4.data(), a.data(), weights4.data(), 1, parameter, nullptr,
                                                      biasOn ? bias.data() : nullptr, nullptr, nullptr);
                    };
                    old();
                    if (h)
                        check(out4.data(), 4, false);
                    out4.check();
                    jobs.push_back({"current_c4", "compute", old, 1});
                }
#endif
                for (int index = 0; index < 3; ++index) {
                    int p = 4 << index;
                    float* w = weights[index]->data();
                    float* out = outputs[index]->data();
                    auto wp = [&, p, w]() { pack(b.data(), w, k, h, p, trans); };
                    auto compute = [&, p, w, out]() {
                        panel(a.data(), w, out, biasOn ? bias.data() : nullptr, e, k, h, p);
                    };
                    auto chain = [&, p, compute, out]() {
                        if (k && e > 1)
                            convert(ac4.data(), a.data(), e, k, 4, k);
                        compute();
                        post(out, scale.data(), shift.data(), slope.data(), e, h, p);
                        if (p != 4)
                            convert(out, c4.data(), e, h, p, 4);
                    };
                    wp();
                    for (int hb = 0; hb < up(h, p) / p; ++hb)
                        for (int z = 0; z < k; ++z)
                            for (int lane = 0; lane < p; ++lane) {
                                int y = hb * p + lane;
                                float ref = y < h ? b[trans ? y * k + z : z * h + y] : 0;
                                if (w[(hb * k + z) * p + lane] != ref)
                                    fail("weight pack mismatch");
                            }
                    compute();
                    if (h)
                        check(out, p, false);
                    // Kernel must leave padded output channels zero before postprocessing.
                    for (int y = h; y < up(h, p); ++y)
                        for (int row = 0; row < e; ++row)
                            if (out[(y / p * e + row) * p + y % p] != 0)
                                fail("output padding mismatch");
                    chain();
                    if (h)
                        check(p == 4 ? out : c4.data(), 4, true);
                    weights[index]->check();
                    outputs[index]->check();
                    c4.check();
                    ac4.check();
                    jobs.push_back({names[index], "weight_pack", wp, 1});
                    jobs.push_back({names[index], "compute", compute, 1});
                    jobs.push_back({names[index], "steady_chain", chain, 1});
                    jobs.push_back({names[index], "first_chain",
                                    [wp, chain]() {
                                        wp();
                                        chain();
                                    },
                                    1});
                }
                if (a != aOriginal || b != bOriginal)
                    fail("input mutated");
                if (correctness || !k || !h)
                    continue;
                for (auto& job : jobs) {
                    while (timeCall(job.fn, job.iterations) * job.iterations < 1000000 && job.iterations < 1048576)
                        job.iterations *= 2;
                }
                // Fresh randomized interleaving each round; every variant gets seven samples.
                for (int round = 0; round < 7; ++round) {
                    std::shuffle(jobs.begin(), jobs.end(), rng);
                    for (auto& job : jobs) {
                        double ns = timeCall(job.fn, job.iterations);
                        sink = raw.data()[0];
                        std::printf("%d,%d,%d,%d,%d,%s,%s,%d,%zu,%.6f\n", e, k, h, trans, biasOn, job.variant,
                                    job.phase, round, job.iterations, ns);
                    }
                }
            }
    std::fprintf(stderr, "checked=%d status=passed timing=%s\n", checked, correctness ? "disabled" : "enabled");
    return 0;
}

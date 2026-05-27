#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifndef ALIMIN
#define ALIMIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef ALIMAX
#define ALIMAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#endif

#ifndef ROUND_UP
#define ROUND_UP(x, y) ((((x) + (y)-1) / (y)) * (y))
#endif

static inline uint64_t perf_next_u64(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline float perf_uniform_float(uint64_t& state, float lo, float hi) {
    const uint32_t bits = static_cast<uint32_t>(perf_next_u64(state) >> 40);
    const float unit = static_cast<float>(bits) / static_cast<float>(0x00ffffffu);
    return lo + (hi - lo) * unit;
}

static inline int perf_uniform_int(uint64_t& state, int lo, int hi) {
    const uint64_t v = perf_next_u64(state);
    return lo + static_cast<int>(v % static_cast<uint64_t>(hi - lo + 1));
}

static inline int8_t perf_uniform_i8(uint64_t& state, int lo = -127, int hi = 127) {
    return static_cast<int8_t>(perf_uniform_int(state, lo, hi));
}

template <typename T>
static inline void perf_touch(const T* data, size_t size) {
    volatile uint64_t acc = 0;
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size * sizeof(T); i += 64) {
        acc += bytes[i];
    }
    if (acc == 0x12345678ULL) {
        std::printf("touch=%llu\n", static_cast<unsigned long long>(acc));
    }
}

static inline int perf_mode(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <mode: 0 scalar, 1 rvv>\n", argv[0]);
        return -1;
    }
    return std::atoi(argv[1]);
}

static inline void perf_keep(float v) {
    volatile float sink = v;
    (void)sink;
}

static inline bool perf_close(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

static inline bool perf_close_int(int a, int b) {
    return a == b;
}

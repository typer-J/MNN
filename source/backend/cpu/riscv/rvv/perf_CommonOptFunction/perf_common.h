#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <sys/types.h>
#include <vector>

#ifndef ALIMIN
#define ALIMIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef ALIMAX
#define ALIMAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef MNN_ASSERT
#define MNN_ASSERT(x)
#endif

struct SumByAxisParams {
    size_t blockNum;
    size_t DST_XUNIT;
    size_t SRC_UNIT;
    size_t unitColBufferSize;
    int oneScale;
    size_t LU;
    size_t valid;
    size_t kernelxy;
    int inputBlock;
};

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

static inline int8_t perf_uniform_i8(uint64_t& state, int lo = -127, int hi = 127) {
    const uint64_t v = perf_next_u64(state);
    return static_cast<int8_t>(lo + static_cast<int>(v % static_cast<uint64_t>(hi - lo + 1)));
}

static inline uint8_t perf_uniform_u8(uint64_t& state) {
    return static_cast<uint8_t>(perf_next_u64(state) & 0xff);
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

static inline void perf_keep_i64(int64_t v) {
    volatile int64_t sink = v;
    (void)sink;
}

static inline bool perf_close(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

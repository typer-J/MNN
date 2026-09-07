// Standalone comparison of exact extracted MNN generic functions and production RVV functions.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

void MNNPackC4(float*, const float*, size_t, size_t, int*);
void MNNUnpackC4(float*, const float*, size_t, size_t, int*);
void MNNScaleAndAddBias(float*, const float*, const float*, const float*, size_t, size_t);
void MNNReluWithSlopeChannel(float*, const float*, const float*, size_t, size_t);
void MNNPackC4_RVV(float*, const float*, size_t, size_t, int*);
void MNNUnpackC4_RVV(float*, const float*, size_t, size_t, int*);
void MNNScaleAndAddBias_RVV(float*, const float*, const float*, const float*, size_t, size_t);
void MNNReluWithSlopeChannel_RVV(float*, const float*, const float*, size_t, size_t);

namespace {
constexpr size_t kGuard = 32;
constexpr uint32_t kCanary = 0x4ad12345;

float fromBits(uint32_t bits) {
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
uint32_t bitsOf(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
struct Guarded {
    explicit Guarded(size_t count) : count(count), values(count + kGuard * 2, fromBits(kCanary)) {}
    float* data() { return values.data() + kGuard; }
    const float* data() const { return values.data() + kGuard; }
    bool guardsOK() const {
        for (size_t i = 0; i < kGuard; ++i) {
            if (bitsOf(values[i]) != kCanary || bitsOf(values[count + kGuard + i]) != kCanary)
                return false;
        }
        return true;
    }
    size_t count;
    std::vector<float> values;
};
struct Random {
    explicit Random(uint32_t seed) : state(seed ? seed : 1) {}
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<float>(static_cast<int>(state % 8193) - 4096) / 1024.0f;
    }
    uint32_t state;
};
enum class Op { Pack, Unpack, Scale, Prelu };
const char* nameOf(Op op) {
    switch (op) {
        case Op::Pack:
            return "PackC4";
        case Op::Unpack:
            return "UnpackC4";
        case Op::Scale:
            return "ScaleAndAddBias";
        case Op::Prelu:
            return "ReluWithSlopeChannel";
    }
    return "unknown";
}
struct Case {
    Op op;
    size_t area;
    size_t channels;
    size_t srcStride;
    size_t dstStride;
    bool inplace;
    size_t quads() const { return (channels + 3) / 4; }
    bool packing() const { return op == Op::Pack || op == Op::Unpack; }
    size_t sourceCount() const {
        if (op == Op::Pack)
            return channels * srcStride;
        if (op == Op::Unpack)
            return quads() * srcStride * 4;
        return quads() * area * 4;
    }
    size_t destinationCount() const {
        if (op == Op::Pack)
            return quads() * dstStride * 4;
        if (op == Op::Unpack)
            return channels * dstStride;
        return sourceCount();
    }
    void json(std::ostream& stream) const {
        stream << "\"op\":\"" << nameOf(op) << "\",\"area\":" << area << ",\"channels\":" << channels
               << ",\"channel_quads\":" << quads() << ",\"src_area_stride\":" << srcStride
               << ",\"dst_area_stride\":" << dstStride << ",\"inplace\":" << (inplace ? "true" : "false");
    }
};
void call(const Case& test, bool rvv, float* dst, const float* src, const float* bias, const float* alpha) {
    int strides[] = {static_cast<int>(test.srcStride), static_cast<int>(test.dstStride)};
    switch (test.op) {
        case Op::Pack:
            (rvv ? MNNPackC4_RVV : MNNPackC4)(dst, src, test.area, test.channels, strides);
            break;
        case Op::Unpack:
            (rvv ? MNNUnpackC4_RVV : MNNUnpackC4)(dst, src, test.area, test.channels, strides);
            break;
        case Op::Scale:
            (rvv ? MNNScaleAndAddBias_RVV : MNNScaleAndAddBias)(dst, src, bias, alpha, test.area, test.quads());
            break;
        case Op::Prelu:
            (rvv ? MNNReluWithSlopeChannel_RVV : MNNReluWithSlopeChannel)(dst, src, alpha, test.area, test.quads());
            break;
    }
}
void fill(Guarded& buffer, Random& random, bool special, size_t phase = 0) {
    const uint32_t boundaries[] = {0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678,
                                   0x00000001, 0x80000001, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff};
    size_t specialIndex = phase;
    for (size_t i = 0; i < buffer.count; ++i) {
        buffer.data()[i] = special && i % 3 != 2 ? fromBits(boundaries[specialIndex++ % 12]) : random.next();
    }
}
// Independent layout/elementwise contract: this checks the extracted baseline as well as the candidate.
void specification(const Case& test, Guarded& expected, const Guarded& src, const Guarded& bias, const Guarded& alpha) {
    if (test.packing()) {
        for (size_t channel = 0; channel < (test.op == Op::Pack ? test.quads() * 4 : test.channels); ++channel) {
            for (size_t x = 0; x < test.area; ++x) {
                if (test.op == Op::Pack) {
                    expected.data()[(channel / 4) * test.dstStride * 4 + x * 4 + channel % 4] =
                        channel < test.channels ? src.data()[channel * test.srcStride + x] : 0.0f;
                } else {
                    expected.data()[channel * test.dstStride + x] =
                        src.data()[(channel / 4) * test.srcStride * 4 + x * 4 + channel % 4];
                }
            }
        }
    } else {
        for (size_t q = 0; q < test.quads(); ++q) {
            for (size_t x = 0; x < test.area; ++x) {
                for (size_t lane = 0; lane < 4; ++lane) {
                    const size_t i = (q * test.area + x) * 4 + lane;
                    const float value = src.data()[i];
                    expected.data()[i] = test.op == Op::Scale
                                             ? value * alpha.data()[q * 4 + lane] + bias.data()[q * 4 + lane]
                                             : (value < 0.0f ? value * alpha.data()[q * 4 + lane] : value);
                }
            }
        }
    }
}
bool identical(const Guarded& left, const Guarded& right, bool nanClass, size_t& bad) {
    if (left.count != right.count || !left.guardsOK() || !right.guardsOK()) {
        bad = std::numeric_limits<size_t>::max();
        return false;
    }
    for (size_t i = 0; i < left.count; ++i) {
        if (nanClass && std::isnan(left.data()[i]) && std::isnan(right.data()[i]))
            continue;
        if (bitsOf(left.data()[i]) != bitsOf(right.data()[i])) {
            bad = i;
            return false;
        }
    }
    return true;
}
bool correctness(const Case& test, uint32_t seed, std::string& detail, size_t& bad) {
    for (int special = 0; special < 2; ++special) {
        Random random(seed);
        Guarded input(test.sourceCount()), bias(test.quads() * 4), alpha(test.quads() * 4);
        fill(input, random, special != 0);
        fill(bias, random, special != 0, 3);
        fill(alpha, random, special != 0, 7);
        const Guarded biasBefore = bias, alphaBefore = alpha;
        Guarded expected(test.destinationCount());
        specification(test, expected, input, bias, alpha);
        Guarded scalar(test.destinationCount()), rvv(test.destinationCount());
        for (int variant = 0; variant < 2; ++variant) {
            Guarded source = input;
            Guarded& result = variant ? rvv : scalar;
            if (test.inplace)
                result = input;
            call(test, variant != 0, result.data(), test.inplace ? result.data() : source.data(), bias.data(),
                 alpha.data());
            if (!identical(source, input, false, bad) || !identical(bias, biasBefore, false, bad) ||
                !identical(alpha, alphaBefore, false, bad)) {
                detail = variant ? "rvv_input_or_parameter_modified" : "scalar_input_or_parameter_modified";
                return false;
            }
            if (!identical(result, expected, !test.packing(), bad)) {
                detail = variant ? "rvv_contract_or_guard_mismatch" : "scalar_contract_or_guard_mismatch";
                detail += special ? "_special" : "_finite";
                return false;
            }
        }
        if (!identical(scalar, rvv, !test.packing(), bad)) {
            detail = "scalar_rvv_mismatch";
            return false;
        }
    }
    return true;
}
double measure(const Case& test, bool rvv, size_t iterations, Guarded& dst, const Guarded& input, const Guarded& bias,
               const Guarded& alpha) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        call(test, rvv, dst.data(), test.inplace ? dst.data() : input.data(), bias.data(), alpha.data());
    }
    return std::chrono::duration<double, std::nano>(Clock::now() - start).count();
}
} // namespace

int main(int argc, char** argv) {
    std::string output;
    int rounds = 7;
    double sampleMs = 1.0;
    uint32_t seed = 20260908;
    size_t vlenBits = 0;
    bool correctnessOnly = false;
    std::string evidenceTier = "unspecified";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--correctness-only")
            correctnessOnly = true;
        else if (i + 1 >= argc)
            return 2;
        else if (arg == "--jsonl")
            output = argv[++i];
        else if (arg == "--rounds")
            rounds = std::atoi(argv[++i]);
        else if (arg == "--sample-ms")
            sampleMs = std::atof(argv[++i]);
        else if (arg == "--seed")
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (arg == "--vlen-bits")
            vlenBits = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--evidence-tier")
            evidenceTier = argv[++i];
        else
            return 2;
    }
    if (output.empty() || rounds < 2 || !std::isfinite(sampleMs) || sampleMs <= 0)
        return 2;
    if (evidenceTier.find_first_not_of("abcdefghijklmnopqrstuvwxyz_") != std::string::npos)
        return 2;
    if (!correctnessOnly && evidenceTier != "target_riscv64_execution") {
        std::fprintf(
            stderr,
            "Timing requires explicit target_riscv64_execution tier; host/emulation uses --correctness-only.\n");
        return 2;
    }
    std::ofstream stream(output.c_str());
    if (!stream)
        return 2;
    stream << std::setprecision(12);
    stream << "{\"record\":\"run\",\"evidence_tier\":\"" << evidenceTier << "\",\"vlen_bits\":" << vlenBits
           << ",\"seed\":" << seed << ",\"correctness_only\":" << (correctnessOnly ? "true" : "false") << "}\n";
    std::vector<size_t> areas = {0, 1, 3, 4, 7, 16, 257, 1024};
    if (vlenBits >= 128) {
        const size_t lanesM8 = vlenBits / 4;
        areas.push_back(lanesM8 - 1);
        areas.push_back(lanesM8);
        areas.push_back(lanesM8 + 1);
        areas.push_back(lanesM8 * 2 + 1);
    }
    std::sort(areas.begin(), areas.end());
    areas.erase(std::unique(areas.begin(), areas.end()), areas.end());
    const size_t channels[] = {0, 1, 3, 4, 5, 7, 8, 9, 13, 32, 64};
    std::vector<Case> cases;
    for (size_t area : areas) {
        for (size_t channel : channels) {
            for (Op op : {Op::Pack, Op::Unpack}) {
                cases.push_back({op, area, channel, area, area, false});
                cases.push_back({op, area, channel, area + 3, area + 7, false});
                cases.push_back({op, area, channel, area + 7, area + 1, false});
            }
            for (Op op : {Op::Scale, Op::Prelu}) {
                for (bool inplace : {false, true})
                    cases.push_back({op, area, channel, area, area, inplace});
            }
        }
    }
    size_t failed = 0, sampleCount = 0;
    volatile uint32_t checksum = 0;
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
        const Case& test = cases[caseIndex];
        std::string detail;
        size_t bad = 0;
        bool passed = correctness(test, seed + static_cast<uint32_t>(caseIndex), detail, bad);
        stream << "{\"record\":\"correctness\",\"case_id\":" << caseIndex << ",";
        test.json(stream);
        stream << ",\"passed\":" << (passed ? "true" : "false") << ",\"variants\":2,\"datasets\":2";
        if (!passed)
            stream << ",\"detail\":\"" << detail << "\",\"bad_index\":" << bad;
        stream << "}\n";
        stream.flush();
        if (!passed) {
            ++failed;
            std::fprintf(stderr, "Case %zu (%s): %s, index=%zu\n", caseIndex, nameOf(test.op), detail.c_str(), bad);
            continue;
        }
        if (correctnessOnly)
            continue;
        Random random(seed + static_cast<uint32_t>(caseIndex));
        Guarded input(test.sourceCount()), bias(test.quads() * 4), alpha(test.quads() * 4);
        fill(input, random, false);
        fill(bias, random, false);
        fill(alpha, random, false);
        // In-place iterations must not drift to Inf/denormals: use an involution or identity.
        if (test.inplace) {
            for (size_t i = 0; i < alpha.count; ++i) {
                alpha.data()[i] = test.op == Op::Scale ? -1.0f : 1.0f;
                bias.data()[i] = 0.0f;
            }
        }
        Guarded dst(test.destinationCount());
        if (test.inplace)
            dst = input;
        for (int warmup = 0; warmup < 4; ++warmup) {
            call(test, false, dst.data(), test.inplace ? dst.data() : input.data(), bias.data(), alpha.data());
            call(test, true, dst.data(), test.inplace ? dst.data() : input.data(), bias.data(), alpha.data());
        }
        size_t iterations = 1;
        while (iterations < (1u << 22)) {
            if (test.inplace)
                dst = input;
            const double scalarElapsed = measure(test, false, iterations, dst, input, bias, alpha);
            if (test.inplace)
                dst = input;
            const double rvvElapsed = measure(test, true, iterations, dst, input, bias, alpha);
            if (std::min(scalarElapsed, rvvElapsed) >= sampleMs * 1e6)
                break;
            iterations *= 2;
        }
        for (int round = 0; round < rounds; ++round) {
            const bool rvvFirst = round % 2 != 0;
            for (int position = 0; position < 2; ++position) {
                const bool rvv = position == 0 ? rvvFirst : !rvvFirst;
                if (test.inplace)
                    dst = input;
                const double elapsed = measure(test, rvv, iterations, dst, input, bias, alpha);
                if (dst.count)
                    checksum ^= bitsOf(dst.data()[dst.count / 2]);
                stream << "{\"record\":\"sample\",\"case_id\":" << caseIndex << ",";
                test.json(stream);
                stream << ",\"variant\":\"" << (rvv ? "rvv" : "mnn_scalar_no_autovec") << "\",\"round\":" << round
                       << ",\"order\":\"" << (rvvFirst ? "BA" : "AB") << "\",\"position\":" << position
                       << ",\"iterations\":" << iterations << ",\"elapsed_ns\":" << elapsed
                       << ",\"ns_per_call\":" << elapsed / iterations << ",\"single_thread\":true,\"timing_input\":\""
                       << (test.inplace ? "stable_involution_or_identity" : "finite_random") << "\"}\n";
                ++sampleCount;
            }
        }
        stream.flush();
    }
    stream << "{\"record\":\"summary\",\"cases\":" << cases.size() << ",\"failed\":" << failed
           << ",\"samples\":" << sampleCount << ",\"checksum\":" << checksum
           << ",\"correctness_only\":" << (correctnessOnly ? "true" : "false") << "}\n";
    stream.flush();
    std::printf("cases=%zu failed=%zu samples=%zu correctness_only=%d\n", cases.size(), failed, sampleCount,
                correctnessOnly);
    return failed == 0 && stream.good() ? 0 : 1;
}

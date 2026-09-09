#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <riscv_vector.h>
#include <sys/syscall.h>
#include <unistd.h>

int main() {
    struct HwprobePair {
        int64_t key;
        uint64_t value;
    } pair = {4, 0};
#ifdef __NR_riscv_hwprobe
    const long probeSyscall = __NR_riscv_hwprobe;
#else
    const long probeSyscall = 258;
#endif
    errno = 0;
    // Same all-online-CPU capability intersection that MNN's CPURuntime queries.
    const long hwprobeResult = syscall(probeSyscall, &pair, size_t(1), size_t(0), nullptr, 0u);
    const int hwprobeErrno = errno;
    const bool mnnRvvCapability = hwprobeResult == 0 && pair.key == 4 && (pair.value & (uint64_t(1) << 2));
    const size_t lanes = __riscv_vsetvlmax_e8m1();
    float input[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    float output[4] = {};
    const size_t vl = __riscv_vsetvl_e32m1(4);
    auto value = __riscv_vle32_v_f32m1(input, vl);
    value = __riscv_vfadd_vf_f32m1(value, 1.0f, vl);
    __riscv_vse32_v_f32m1(output, value, vl);
    const bool valid = vl == 4 && output[0] == 2.0f && output[1] == -1.0f && output[2] == 4.0f && output[3] == 5.0f;
    std::printf(
        "{\"vlen_bits\":%zu,\"vector_fp32_probe_passed\":%s,\"hwprobe_return\":%ld,"
        "\"hwprobe_errno\":%d,\"hwprobe_key\":%lld,\"hwprobe_ima_ext_0\":%llu,"
        "\"mnn_rvv_capability_detected\":%s}\n",
        lanes * 8, valid ? "true" : "false", hwprobeResult, hwprobeErrno, static_cast<long long>(pair.key),
        static_cast<unsigned long long>(pair.value), mnnRvvCapability ? "true" : "false");
    return valid ? 0 : 1;
}

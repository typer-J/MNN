#!/usr/bin/env bash
# Run from an isolated candidate checkout. No installation or remote connection is performed.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: bash benchmark/rvv_scalar_compare/sg2044_validate.sh [NEW_OUTPUT_DIRECTORY]

Required environment:
  ORIGINAL_REPO   Original repository to protect; this checkout must be outside it.
  GCC14_PREFIX    Compiler prefix containing bin/g++ and bin/gcc; alternatively
                 set CXX to one compiler executable and optionally CC explicitly.
Optional environment:
  PYTHON=python3          Existing Python interpreter; no packages are installed.
  CPUSET                 Comma-separated allowed logical CPUs; default first allowed CPU.
  EXPECT_VLEN=512        Expected VLEN in bits. The default is a hypothesis, not
                        verified SG2044 identity; a measured mismatch stops the run.
  PERF_PROCESSES=3       Fresh performance processes after correctness passes.
  BENCH_ROUNDS=7         AB/BA rounds per case in each performance process.
  BENCH_SAMPLE_MS=1      Target calibration duration of the faster implementation.
  SKIP_INTEGRATION=0     Set 1 to skip full MNN integration explicitly.
  JOBS=4                Passed to integration.sh for its build parallelism.

Outputs default to benchmark/rvv_scalar_compare/results/session-<UTC> within
this checkout. Existing output directories and locations outside this checkout
are rejected. Checks and failures retain logs, exit codes and summary.json;
the final tar.gz and SHA256 exclude integration/build-scalar and build-rvv.
ISA/VLEN execution alone does not establish physical SG2044 identity or actual
MNN CoreFunctions dispatch. Review the recorded current load/frequency context.
USAGE
    exit 0
fi
if [[ $# -gt 1 ]]; then
    printf '%s\n' 'Expected at most one fresh output directory; use --help.' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${PYTHON:-python3}" - "$script_dir" "${1:-}" <<'PY'
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

here = Path(sys.argv[1]).resolve()
root = here.parent.parent
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
out = (Path(sys.argv[2]) if sys.argv[2] else here / "results" / ("session-" + stamp)).resolve()


def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


# Do not create even a failure directory until its location is proven safe.
original_value = os.environ.get("ORIGINAL_REPO")
if not original_value:
    sys.exit("ORIGINAL_REPO is required to protect the original checkout; no files were created.")
original = Path(original_value).resolve()
if not original.is_dir() or within(root, original):
    sys.exit("The original repository must exist and this candidate must be outside it; no files were created.")
archive_path = out.with_name(out.name + ".tar.gz")
if (out == root or not within(out, root) or within(out, original) or out.exists()
        or archive_path.exists() or archive_path.with_name(archive_path.name + ".sha256").exists()):
    sys.exit("Output must be a fresh directory inside the isolated candidate checkout; no files were created.")
out.mkdir(parents=True)
summary = {"status": "running", "evidence_tier": "preflight_only", "utc": stamp,
           "physical_board_identity": "requires external verification; uname/VLEN cannot rule out emulation",
           "dispatch_validation": "microbench uses direct calls; hwprobe alone does not prove CoreFunctions selection",
           "isolation_check": "passed", "integration": "not_run", "stages": [],
           "limitations": ["No model-level correctness or end-to-end model performance acceptance is performed here"]}
selected = []
exit_code = 1


def save():
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def stage(name, command, env=None, pinned=False):
    start = time.monotonic()
    record = {"name": name, "status": "running"}
    summary["stages"].append(record)
    save()
    # Results are local evidence; the script itself contains no site-specific paths.
    (out / (name + ".command.json")).write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    def pin():
        os.sched_setaffinity(0, selected)
    print("Running " + name, flush=True)
    try:
        with (out / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace",
                                       preexec_fn=pin if pinned else None)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            result = process.wait()
        record["exit_code"] = result
        (out / (name + ".exitcode")).write_text(str(result) + "\n")
        record["status"] = "passed" if result == 0 else "failed"
        if result:
            raise RuntimeError(name + " failed with exit code " + str(result))
    except Exception:
        record["status"] = "failed"
        record.setdefault("exit_code", 127)
        (out / (name + ".exitcode")).write_text(str(record["exit_code"]) + "\n")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        save()


def context(label):
    data = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "machine": platform.machine(), "kernel_release": platform.release(),
            "online_cpu_count": os.cpu_count(), "load_average": list(os.getloadavg()),
            "allowed_cpus": sorted(os.sched_getaffinity(0)), "selected_cpus": selected}
    data["cpu_information"] = [line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                               if line.split(":", 1)[0].strip() in
                               {"processor", "hart", "isa", "uarch", "mmu", "mvendorid", "marchid", "mimpid"}]
    data["frequency"] = {}
    for cpu in selected:
        base = Path("/sys/devices/system/cpu") / ("cpu" + str(cpu)) / "cpufreq"
        values = {}
        for field in ["scaling_cur_freq", "cpuinfo_max_freq", "cpuinfo_min_freq", "scaling_governor"]:
            try:
                values[field] = (base / field).read_text().strip()
            except OSError:
                values[field] = None
        data["frequency"][str(cpu)] = values
    data["thermal"] = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            data["thermal"][zone.name] = {"type": (zone / "type").read_text().strip(),
                                          "temperature": (zone / "temp").read_text().strip()}
        except OSError:
            pass
    (out / (label + ".context.json")).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


try:
    save()
    machine = platform.machine().lower()
    summary["observed_machine"] = machine
    if platform.system() != "Linux" or machine != "riscv64":
        raise RuntimeError("This entrypoint requires Linux riscv64; refusing host/emulation-labelled performance evidence")
    summary["evidence_tier"] = "target_execution_attempt"
    allowed = set(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("The process has no allowed CPUs")
    requested = os.environ.get("CPUSET")
    selected = sorted(set(map(int, requested.split(",")))) if requested else [min(allowed)]
    if not selected or not set(selected).issubset(allowed):
        raise RuntimeError("CPUSET must contain only currently allowed logical CPU IDs")
    summary["affinity_requested"] = requested or "first_allowed_cpu"
    summary["affinity_selected"] = selected
    context("preflight")
    vlen = int(os.environ.get("EXPECT_VLEN", "512"))
    processes = int(os.environ.get("PERF_PROCESSES", "3"))
    rounds = int(os.environ.get("BENCH_ROUNDS", "7"))
    sample_ms = float(os.environ.get("BENCH_SAMPLE_MS", "1"))
    skip_integration = os.environ.get("SKIP_INTEGRATION", "0")
    if (vlen < 128 or vlen % 8 or processes < 1 or rounds < 2 or not math.isfinite(sample_ms)
            or sample_ms <= 0 or skip_integration not in {"0", "1"}):
        raise RuntimeError("Invalid EXPECT_VLEN/PERF_PROCESSES/BENCH_ROUNDS/BENCH_SAMPLE_MS/SKIP_INTEGRATION")
    summary["expected_vlen_bits"] = vlen
    summary["vlen_expectation_source"] = "caller" if "EXPECT_VLEN" in os.environ else "unverified_default_512"
    summary["performance_processes_requested"] = processes
    if os.environ.get("CXX"):
        compiler = shutil.which(os.environ["CXX"])
    else:
        prefix = os.environ.get("GCC14_PREFIX")
        compiler = str(Path(prefix) / "bin/g++") if prefix else None
    if not compiler or not Path(compiler).is_file() or not os.access(compiler, os.X_OK):
        raise RuntimeError("Set GCC14_PREFIX with an executable bin/g++, or CXX to one executable (no flags)")
    compiler = str(Path(compiler).resolve())
    cc_name = Path(compiler).name.replace("g++", "gcc")
    cc = shutil.which(os.environ["CC"]) if os.environ.get("CC") else str(Path(compiler).with_name(cc_name))
    if not cc or not Path(cc).is_file() or not os.access(cc, os.X_OK) or cc == compiler:
        raise RuntimeError("A matching C compiler is required in bin/gcc, or set CC explicitly")
    cc = str(Path(cc).resolve())
    for label, executable in [("cxx", compiler), ("cc", cc)]:
        stage(label + "-version", [executable, "--version"])
        stage(label + "-major", [executable, "-dumpfullversion", "-dumpversion"])
        stage(label + "-target", [executable, "-dumpmachine"])
        version = (out / (label + "-major.log")).read_text().strip()
        target = (out / (label + "-target.log")).read_text().strip()
        if version.split(".")[0] != "14" or not target.startswith("riscv64"):
            raise RuntimeError(label + " must be GCC major 14 targeting riscv64")
    preflight = out / "preflight"
    preflight.mkdir()
    memory_source = preflight / "memory_thread_probe.cpp"
    memory_source.write_text('''#include <memory>
#include <thread>
#include <atomic>
int main() {
    auto value = std::make_shared<std::atomic<int>>(0);
    std::thread worker([value] { value->store(42); });
    worker.join();
    return value->load() == 42 ? 0 : 1;
}
''')
    common_flags = ["-O2", "-std=c++11", "-march=rv64gc", "-mabi=lp64d", "-fno-fast-math", "-ffp-contract=off"]
    stage("memory-build", [compiler] + common_flags + [str(memory_source), "-pthread", "-o", str(preflight / "memory-probe")])
    stage("memory-run", [str(preflight / "memory-probe")], pinned=True)
    stage("rvv-probe-build", [compiler] + common_flags + ["-march=rv64gcv", str(here / "probe.cpp"),
                                                          "-o", str(preflight / "rvv-probe")])
    stage("rvv-probe-run", [str(preflight / "rvv-probe")], pinned=True)
    probe = json.loads((out / "rvv-probe-run.log").read_text().strip())
    summary["probe"] = probe
    if not probe.get("vector_fp32_probe_passed") or probe.get("vlen_bits") != vlen:
        raise RuntimeError("RVV probe/expected VLEN mismatch; verify the actual target before changing EXPECT_VLEN")
    summary["evidence_tier"] = "target_riscv64_execution"
    if not probe.get("mnn_rvv_capability_detected"):
        summary["capability_note"] = "RVV executes but MNN-style hwprobe does not report V; normal MNN dispatch may fall back"
    runner_env = os.environ.copy()
    runner_env["CXX"] = shlex.quote(compiler)
    cpuset = ",".join(map(str, selected))
    arguments = [sys.executable, str(here / "run.py"), "--cpus", cpuset, "--expect-vlen", str(vlen),
                 "--rounds", str(rounds), "--sample-ms", str(sample_ms)]
    stage("correctness", arguments + ["--correctness-only", "--out", str(out / "correctness")], env=runner_env)
    for index in range(1, processes + 1):
        label = "performance-" + str(index)
        context(label + "-before")
        stage(label, arguments + ["--out", str(out / label)], env=runner_env)
        context(label + "-after")
    if skip_integration == "1":
        summary["integration"] = "explicitly_skipped"
    else:
        integration = here / "integration.sh"
        if not integration.is_file():
            raise RuntimeError("integration.sh is missing; full validation is incomplete")
        integration_env = os.environ.copy()
        integration_env.update({"CC": cc, "CXX": compiler, "CPUSET": cpuset, "PYTHON": sys.executable})
        summary["integration"] = "running"
        stage("integration", ["bash", str(integration), str(out / "integration")], env=integration_env)
        integration_summary = out / "integration/summary.json"
        if not integration_summary.is_file():
            raise RuntimeError("Integration exited successfully without summary.json; results are incomplete")
        summary["integration_summary"] = json.loads(integration_summary.read_text())
        summary["integration"] = "passed"
    summary["status"] = "passed"
    exit_code = 0
except Exception as error:
    summary["status"] = "failed"
    summary["error"] = str(error)
    if summary["integration"] == "running":
        summary["integration"] = "failed"
        integration_summary = out / "integration/summary.json"
        if integration_summary.is_file():
            try:
                summary["integration_summary"] = json.loads(integration_summary.read_text())
            except (ValueError, OSError):
                summary["limitations"].append("The failed integration summary could not be read")
    (out / "failure.log").write_text(str(error) + "\n", encoding="utf-8")
    print("FAILED: " + str(error), file=sys.stderr, flush=True)
finally:
    summary["exit_code"] = exit_code
    save()
    (out / "exitcode").write_text(str(exit_code) + "\n")
    # Archive results only, never walk the checkout or follow symlinks outside results.
    archive = out.with_name(out.name + ".tar.gz")
    excluded = {out / "integration/build-scalar", out / "integration/build-rvv"}
    try:
        with tarfile.open(archive, "w:gz") as tar:
            for directory, subdirs, files in os.walk(out, followlinks=False):
                directory = Path(directory)
                subdirs[:] = [name for name in subdirs if directory / name not in excluded
                              and not (directory / name).is_symlink()]
                for name in files:
                    path = directory / name
                    if not path.is_symlink():
                        tar.add(path, arcname=str(Path(out.name) / path.relative_to(out)), recursive=False)
        sha = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        checksum = archive.with_name(archive.name + ".sha256")
        checksum.write_text(sha.hexdigest() + "  " + archive.name + "\n")
        print("Archive: " + str(archive), flush=True)
        print("SHA256: " + str(checksum), flush=True)
    except Exception as error:
        summary["archive_error"] = str(error)
        summary["status"] = "failed"
        summary["exit_code"] = exit_code = 1
        save()
        (out / "exitcode").write_text("1\n")
        print("Archive failed: " + str(error), file=sys.stderr, flush=True)
    print("Results: " + str(out), flush=True)
    print("Status: " + summary["status"] + "; integration=" + summary["integration"], flush=True)
sys.exit(exit_code)
PY

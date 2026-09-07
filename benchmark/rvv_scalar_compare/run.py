#!/usr/bin/env python3
"""Build exact MNN scalar source and real RVV translation units in one benchmark."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
COMMON = "source/backend/cpu/compute/CommonOptFunction.cpp"
NAMES = ["MNNPackC4", "MNNUnpackC4", "MNNScaleAndAddBias", "MNNReluWithSlopeChannel"]
RVV = "source/backend/cpu/riscv/rvv/"
DEPENDENCIES = ["source/math/Vec.hpp", "source/core/Macro.h", "source/core/SimdHeader.h",
                "include/MNN/MNNDefine.h"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def lexical_mask(text):
    # Preserve offsets while hiding braces in comments, string and character literals.
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda match: " " * len(match.group()), text)


def extract_function(text, name, template=False):
    mask = lexical_mask(text)
    expression = (r"template\s*<\s*typename\s+T\s*>\s*" if template else "")
    expression += r"void\s+" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{"
    matches = list(re.finditer(expression, mask))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one definition for " + name + ", got " + str(len(matches)))
    begin = matches[0].start()
    opened = matches[0].end() - 1
    depth = 1
    end = opened + 1
    while end < len(mask) and depth:
        depth += (mask[end] == "{") - (mask[end] == "}")
        end += 1
    if depth:
        raise RuntimeError("Unbalanced definition for " + name)
    fragment = text[begin:end]
    return fragment, {"name": name, "line": text.count("\n", 0, begin) + 1,
                      "sha256_utf8": digest(fragment.encode("utf-8"))}


def extract_scalar(snapshot):
    text = (snapshot / COMMON).read_bytes().decode("utf-8")
    fragments, provenance = [], []
    for name in ["MNNPackC4Common", "MNNUnpackC4Common"] + NAMES:
        fragment, record = extract_function(text, name, name.endswith("Common"))
        fragments.append(fragment)
        provenance.append(record)
    prefix = '#include <cstddef>\n#include "math/Vec.hpp"\nusing Vec4 = MNN::Math::Vec<float, 4>;\n'
    # Definitions are byte-for-byte UTF-8 fragments from the tracked MNN source.
    return prefix + "\n\n".join(fragments) + "\n", provenance


def run_command(command, log, cwd=None):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", cwd=cwd)
    log.write(result.stdout)
    log.flush()
    if result.returncode:
        raise RuntimeError("Command failed (exit " + str(result.returncode) + "): " + Path(command[0]).name)
    return result.stdout


def run_benchmark(command, recorded_command, log, out, metadata, save_metadata):
    metadata["benchmark_command"] = recorded_command
    save_metadata()
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    metadata["benchmark_exit_code"] = result.returncode
    description = "exit code " + str(result.returncode)
    if os.name == "posix" and result.returncode < 0:
        number = -result.returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = "unknown"
        metadata["benchmark_signal"] = {"number": number, "name": name}
        description += ", signal " + name + " (" + str(number) + ")"
    metadata["benchmark_exit_description"] = description
    log.flush()
    # Preserve the real process outcome before reading or aggregating any artifacts.
    save_metadata()
    results_path = out / "results.jsonl"
    if not results_path.is_file():
        log_path = out / "run.log"
        try:
            with log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 16000))
                tail = stream.read().decode("utf-8", errors="replace")[-4000:]
        except OSError as error:
            tail = "[run.log unavailable: " + str(error) + "]"
        metadata["status"] = "failed"
        metadata["benchmark_log_tail"] = tail
        metadata["benchmark_missing_artifact"] = "results.jsonl"
        save_metadata()
        raise RuntimeError("Benchmark did not produce results.jsonl (" + description + "). "
                           "Command: " + json.dumps(recorded_command) + ". "
                           "Evidence: " + str(out / "metadata.json") + "; log: " + str(log_path) +
                           ". run.log tail (up to 4000 characters):\n" + (tail or "[empty]"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are rejected")
    parser.add_argument("--expect-vlen", type=int, default=int(os.environ.get("EXPECT_VLEN", "0")),
                        help="Expected VLEN in bits; zero records probe without assuming a chip identity")
    parser.add_argument("--cpus", "--cpu", default=os.environ.get("CPUSET"),
                        help="Optional comma-separated logical CPU IDs; applied before probe and benchmark")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--extract-only", action="store_true", help="Snapshot and extract source without compiling")
    parser.add_argument("--correctness-only", action="store_true", help="Run all correctness cases without timing")
    parser.add_argument("--host-scalar-check", action="store_true",
                        help="Test harness and scalar contract locally; NO RVV execution or performance evidence")
    args = parser.parse_args()
    if args.rounds < 2 or not math.isfinite(args.sample_ms) or args.sample_ms <= 0 or args.expect_vlen < 0:
        parser.error("rounds must be >=2, sample-ms finite and >0, expect-vlen >=0")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.out or HERE / "results" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "status": "started", "utc": stamp,
                "evidence_tier": "source_extraction" if args.extract_only else
                ("host_scalar_harness_check" if args.host_scalar_check else "target_riscv64_execution"),
                "machine": platform.machine(), "kernel_release": platform.release(),
                "expected_vlen_bits": args.expect_vlen or None, "rounds": args.rounds,
                "sample_ms": args.sample_ms, "seed": args.seed,
                "correctness_only": args.correctness_only or args.host_scalar_check,
                "baseline": "Exact MNN generic function bodies, GCC auto-vectorization disabled",
                "float_semantics": "-fno-fast-math -ffp-contract=off on both variants",
                "dispatch_validation": "Direct kernel calls only; hwprobe observation does not prove MNN CoreFunctions selection",
                "hardware_identity": "Physical board identity and absence of emulation require external verification",
                "source_hashes": {}, "object_hashes": {}, "commands": []}
    def save_metadata():
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_metadata()
    try:
        scoped = [COMMON, "source/backend/cpu/CPURuntime.cpp"] + DEPENDENCIES + [RVV + name + ".cpp" for name in NAMES]
        scoped += [RVV + "MNNRvvC4Functions.hpp"]
        scoped += [str(path.relative_to(ROOT)).replace("\\", "/") for path in
                   [HERE / "run.py", HERE / "benchmark.cpp", HERE / "probe.cpp"]]
        snapshot = out / "source_snapshot"
        for relative in scoped:
            data = (ROOT / relative).read_bytes()
            metadata["source_hashes"][relative] = digest(data)
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        scalar, fragments = extract_scalar(snapshot)
        (out / "scalar.cpp").write_bytes(scalar.encode("utf-8"))
        metadata["scalar_fragments"] = fragments
        metadata["generated_scalar_sha256"] = digest(scalar.encode("utf-8"))
        metadata["git_head"] = None
        metadata["git_scoped_status"] = None
        git = shutil.which("git")
        if git:
            head = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            if head.returncode == 0:
                metadata["git_head"] = head.stdout.strip()
                status = subprocess.run([git, "status", "--porcelain", "--"] + scoped,
                                        cwd=ROOT, capture_output=True, text=True)
                if status.returncode == 0:
                    metadata["git_scoped_status"] = status.stdout.splitlines()
        if args.extract_only:
            metadata["status"] = "extracted"
            save_metadata()
            print("Extracted exact source into " + str(out))
            return 0
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        if not compiler:
            raise RuntimeError("CXX is empty")
        resolved_compiler = shutil.which(compiler[0])
        if not resolved_compiler:
            raise RuntimeError("CXX executable not found")
        # This identifies the invoked driver/wrapper, not an assumed compiler behind a wrapper.
        metadata["invoked_executable_sha256"] = digest(Path(resolved_compiler).read_bytes())
        metadata["invoked_executable_name"] = Path(resolved_compiler).name
        def public_command(command):
            result = [Path(command[0]).name]
            for argument in command[1:]:
                result.append(str(argument).replace(str(out), "<RESULT>").replace(str(ROOT), "<REPO>")
                              .replace(str(Path.home()), "<USER_HOME>"))
            return result
        metadata["compiler_invocation"] = public_command(compiler)
        with (out / "build.log").open("w", encoding="utf-8") as log:
            version = run_command(compiler + ["--version"], log).splitlines()[0]
            metadata["compiler_version"] = version
            metadata["compiler_target"] = run_command(compiler + ["-dumpmachine"], log).strip()
            if not args.host_scalar_check and not metadata["compiler_target"].startswith("riscv64"):
                raise RuntimeError("Target run requires a riscv64 compiler; use --host-scalar-check for host checks")
            # Current MSVC standard-library headers require C++14 even for this C++11 harness.
            standard = "c++14" if args.host_scalar_check and "msvc" in metadata["compiler_target"] else "c++11"
            metadata["language_standard"] = standard
            flags = ["-O3", "-std=" + standard, "-fno-exceptions", "-fno-rtti", "-fno-fast-math",
                     "-ffp-contract=off", "-fno-lto", "-I", str(snapshot / "source"),
                     "-I", str(snapshot / "include")]
            if not args.host_scalar_check:
                flags += ["-march=rv64gcv", "-mabi=lp64d"]
            scalar_flags = (["-fno-vectorize", "-fno-slp-vectorize"] if "clang" in version.lower()
                            else ["-fno-tree-vectorize", "-fno-tree-slp-vectorize"])
            if not args.host_scalar_check:
                scalar_flags += ["-march=rv64gc"]
            metadata["scalar_extra_flags"] = scalar_flags
            def build(source, destination, extra=None, link=False):
                command = compiler + flags + (extra or []) + ([] if link else ["-c"])
                command += [str(source), "-o", str(destination)]
                # Reproducible arguments with local source/result paths made portable.
                metadata["commands"].append(public_command(command))
                run_command(command, log)
            build(out / "scalar.cpp", out / "scalar.o", scalar_flags)
            objects = [out / "scalar.o"]
            if args.host_scalar_check:
                # A deliberate scalar alias exercises harness contracts; it is never labelled RVV evidence.
                aliases = scalar
                for name in NAMES:
                    aliases = re.sub(r"\b" + name + r"\b", name + "_RVV", aliases)
                (out / "host_alias.cpp").write_text(aliases, encoding="utf-8")
                build(out / "host_alias.cpp", out / "host_alias.o", scalar_flags)
                objects.append(out / "host_alias.o")
            else:
                for name in NAMES:
                    obj = out / (name + "_RVV.o")
                    build(snapshot / (RVV + name + ".cpp"), obj)
                    objects.append(obj)
                build(snapshot / "benchmark/rvv_scalar_compare/probe.cpp", out / "probe", link=True)
            build(snapshot / "benchmark/rvv_scalar_compare/benchmark.cpp", out / "benchmark.o")
            objects.append(out / "benchmark.o")
            binary = out / ("compare.exe" if os.name == "nt" else "compare")
            link_command = compiler + [str(x) for x in objects] + ["-o", str(binary), "-pthread"]
            metadata["commands"].append(public_command(link_command))
            run_command(link_command, log)
        metadata["binary_sha256"] = digest(binary.read_bytes())
        for obj in objects:
            metadata["object_hashes"][obj.name] = digest(obj.read_bytes())
        if hasattr(os, "sched_getaffinity"):
            metadata["affinity_inherited"] = sorted(os.sched_getaffinity(0))
        if args.cpus:
            requested = set(map(int, args.cpus.split(",")))
            metadata["affinity_requested"] = sorted(requested)
            os.sched_setaffinity(0, requested)
            if os.sched_getaffinity(0) != requested:
                raise RuntimeError("Effective affinity differs from requested CPU set")
        if hasattr(os, "sched_getaffinity"):
            metadata["affinity_effective"] = sorted(os.sched_getaffinity(0))
        def check_sources():
            changed = [name for name, sha in metadata["source_hashes"].items()
                       if digest((ROOT / name).read_bytes()) != sha]
            metadata["changed_source_files"] = changed
            if changed:
                raise RuntimeError("Source changed during run; preserve this result as superseded and rerun")
        check_sources()
        with (out / "run.log").open("w", encoding="utf-8") as log:
            if not args.host_scalar_check:
                if platform.machine().lower() != "riscv64":
                    raise RuntimeError("Target execution requires riscv64; host checks use --host-scalar-check")
                probe = json.loads(run_command([str(out / "probe")], log).strip())
                metadata["probe"] = probe
                if not probe["vector_fp32_probe_passed"]:
                    raise RuntimeError("RVV FP32 probe failed")
                if args.expect_vlen and probe["vlen_bits"] != args.expect_vlen:
                    raise RuntimeError("Measured VLEN does not match EXPECT_VLEN")
                metadata["vlen_expectation_check"] = "passed" if args.expect_vlen else "not_requested"
            else:
                metadata["probe"] = None
            save_metadata()
            command = [str(binary), "--jsonl", str(out / "results.jsonl"), "--rounds", str(args.rounds),
                       "--sample-ms", str(args.sample_ms), "--seed", str(args.seed),
                       "--evidence-tier", metadata["evidence_tier"]]
            if args.host_scalar_check or args.correctness_only:
                command += ["--correctness-only"]
            if metadata.get("probe"):
                command += ["--vlen-bits", str(metadata["probe"]["vlen_bits"])]
            result = run_benchmark(command, public_command(command), log, out, metadata, save_metadata)
        records = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()]
        samples = [row for row in records if row["record"] == "sample"]
        if samples:
            with (out / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
            comparisons = []
            by_case = {}
            for sample in samples:
                by_case.setdefault(sample["case_id"], []).append(sample)
            for case_id, rows in sorted(by_case.items()):
                record = {key: rows[0][key] for key in ["case_id", "op", "area", "channels", "channel_quads",
                                                       "src_area_stride", "dst_area_stride", "inplace", "timing_input"]}
                for variant, label in [("mnn_scalar_no_autovec", "scalar"), ("rvv", "rvv")]:
                    values = [row["ns_per_call"] for row in rows if row["variant"] == variant]
                    median = statistics.median(values)
                    record[label + "_samples"] = len(values)
                    record[label + "_median_ns"] = median
                    record[label + "_min_ns"] = min(values)
                    record[label + "_max_ns"] = max(values)
                    record[label + "_mad_ns"] = statistics.median(abs(value - median) for value in values)
                meaningful = record["area"] > 0 and record["channels"] > 0 and record["rvv_median_ns"] > 0
                record["speedup_scalar_over_rvv"] = record["scalar_median_ns"] / record["rvv_median_ns"] if meaningful else None
                comparisons.append(record)
            with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
        check_sources()
        metadata["summary"] = next((row for row in reversed(records) if row["record"] == "summary"), None)
        if metadata["summary"] is None or metadata["summary"].get("cases", 0) == 0:
            raise RuntimeError("Benchmark did not produce a nonempty completion summary (" +
                               metadata["benchmark_exit_description"] + "); evidence: " +
                               str(out / "metadata.json") + "; log: " + str(out / "run.log"))
        metadata["status"] = "passed" if result.returncode == 0 and metadata["summary"].get("failed", 0) == 0 else "failed"
        save_metadata()
        print(json.dumps({"status": metadata["status"], "evidence_tier": metadata["evidence_tier"],
                          "summary": metadata["summary"], "output": str(out)}))
        return 0 if metadata["status"] == "passed" else 1
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        save_metadata()
        print("FAILED: " + str(error) + "; retained records: " + str(out), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

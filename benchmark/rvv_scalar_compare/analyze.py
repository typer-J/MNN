#!/usr/bin/env python3
"""Validate and summarize a completed RVV/scalar session without executing its files.

Usage: python analyze.py SESSION --output NEW_DIRECTORY [--archive SESSION.tar.gz]
The optional archive is checked against its adjacent .sha256 (or --checksum).
Only the four-function, three-process schema produced by this benchmark is supported.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


RUNS = ("correctness", "performance-1", "performance-2", "performance-3")
VARIANTS = ("mnn_scalar_no_autovec", "rvv")
KEY_FIELDS = ("case_id", "op", "area", "channels", "channel_quads",
              "src_area_stride", "dst_area_stride", "inplace")
OPS = ("PackC4", "UnpackC4", "ScaleAndAddBias", "ReluWithSlopeChannel")
SOURCES = {
    "source/backend/cpu/compute/CommonOptFunction.cpp",
    "source/backend/cpu/CPURuntime.cpp", "source/math/Vec.hpp", "source/core/Macro.h",
    "source/core/SimdHeader.h", "include/MNN/MNNDefine.h",
    "source/backend/cpu/riscv/rvv/MNNRvvC4Functions.hpp",
    "benchmark/rvv_scalar_compare/run.py", "benchmark/rvv_scalar_compare/benchmark.cpp",
    "benchmark/rvv_scalar_compare/probe.cpp",
} | {"source/backend/cpu/riscv/rvv/MNN" + op + ".cpp" for op in OPS}
OBJECTS = {"scalar.o", "benchmark.o"} | {"MNN" + op + "_RVV.o" for op in OPS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Evidence:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.hashes = {}

    def data(self, relative):
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
                "Unsafe evidence path: " + relative)
        target = self.root.joinpath(*path.parts)
        try:
            target.resolve(strict=True).relative_to(self.root)
        except ValueError:
            raise ValueError("Evidence escapes session")
        require(not any(self.root.joinpath(*path.parts[:i]).is_symlink()
                        for i in range(1, len(path.parts) + 1)), "Symlink evidence is not supported")
        content = target.read_bytes()
        self.hashes[relative] = digest(content)
        return content

    def text(self, relative):
        return self.data(relative).decode("utf-8")

    def json(self, relative):
        return json.loads(self.text(relative))

    def csv(self, relative):
        rows = list(csv.DictReader(io.StringIO(self.text(relative))))
        require(rows and all(None not in row and None not in row.values() for row in rows),
                "Malformed/empty CSV: " + relative)
        return rows

    def verify_hash(self, relative, expected):
        require(digest(self.data(relative)) == expected, "SHA256 mismatch: " + relative)


def number_equal(actual, expected, context):
    value = float(actual)
    require(math.isfinite(value) and math.isfinite(expected)
            and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-9),
            "Numeric mismatch: " + context)


def compare_csv(row, expected, context):
    require(set(row) == set(expected), "CSV fields mismatch: " + context)
    for name, value in expected.items():
        actual = row[name]
        if value is None:
            require(actual == "", "Expected empty value: " + context + "/" + name)
        elif isinstance(value, float):
            number_equal(actual, value, context + "/" + name)
        else:
            require(actual == str(value), "CSV mismatch: " + context + "/" + name)


def case_key(row):
    require(all(name in row for name in KEY_FIELDS), "Missing case key")
    require(type(row["inplace"]) is bool and row["op"] in OPS, "Invalid case type/op")
    for name in KEY_FIELDS:
        if name not in ("inplace", "op"):
            require(type(row[name]) is int and row[name] >= 0, "Invalid case dimension: " + name)
    require(row["channel_quads"] == (row["channels"] + 3) // 4, "Channel quad mismatch")
    return tuple(row[name] for name in KEY_FIELDS)


def validate_build(meta, evidence, run):
    require(set(meta["source_hashes"]) == SOURCES, "Unexpected source set: " + run)
    require(set(meta["object_hashes"]) == OBJECTS, "Unexpected object set: " + run)
    for name, expected in meta["source_hashes"].items():
        evidence.verify_hash(run + "/source_snapshot/" + name, expected)
    for name, expected in meta["object_hashes"].items():
        evidence.verify_hash(run + "/" + name, expected)
    evidence.verify_hash(run + "/scalar.cpp", meta["generated_scalar_sha256"])
    evidence.verify_hash(run + "/compare", meta["binary_sha256"])
    require(meta["language_standard"] == "c++11", "Unexpected C++ standard")
    require("14." in meta["compiler_version"] and meta["compiler_target"].startswith("riscv64-"),
            "Expected recorded GCC 14 RISC-V compiler")
    outputs = {}
    for command in meta["commands"]:
        require(isinstance(command, list) and command.count("-o") == 1, "Invalid compiler command")
        output = command[command.index("-o") + 1].rsplit("/", 1)[-1]
        require(output not in outputs, "Duplicate compiler output")
        outputs[output] = command
        require(not any(flag == "-Ofast" or flag.startswith("-flto") or flag == "-ffast-math"
                        or flag == "-funsafe-math-optimizations" for flag in command),
                "Incompatible optimization flag")
        if output != "compare":
            required = {"-O3", "-std=c++11", "-fno-fast-math", "-ffp-contract=off", "-fno-lto",
                        "-fno-exceptions", "-fno-rtti", "-mabi=lp64d"}
            require(required.issubset(command), "Missing benchmark compiler flags: " + output)
            allowed_float_flags = {"-fno-fast-math", "-ffp-contract=off", "-fno-lto", "-fno-exceptions", "-fno-rtti"}
            if output == "scalar.o":
                allowed_float_flags |= {"-fno-tree-vectorize", "-fno-tree-slp-vectorize"}
            require(all(not flag.startswith("-f") or flag in allowed_float_flags for flag in command),
                    "Unexpected/overriding compiler flag: " + output)
            require(all(not flag.startswith("-O") or flag == "-O3" for flag in command),
                    "Overriding optimization level: " + output)
            arch = [flag for flag in command if flag.startswith("-march=")]
            require(arch and arch[-1] == ("-march=rv64gc" if output == "scalar.o" else "-march=rv64gcv"),
                    "Effective ISA mismatch: " + output)
    require(set(outputs) == OBJECTS | {"probe", "compare"}, "Incomplete build commands")
    require({"-fno-tree-vectorize", "-fno-tree-slp-vectorize"}.issubset(outputs["scalar.o"]),
            "Scalar auto-vectorization was not disabled")


def validate_run(evidence, run, vlen):
    meta = evidence.json(run + "/metadata.json")
    require(meta["schema_version"] == 1, "Unsupported metadata schema")
    require(meta["status"] == "passed" and meta["benchmark_exit_code"] == 0,
            "Run did not pass: " + run)
    require(meta["evidence_tier"] == "target_riscv64_execution" and meta["machine"] == "riscv64",
            "Not target RISC-V evidence: " + run)
    require(meta["expected_vlen_bits"] == vlen and meta["probe"]["vlen_bits"] == vlen
            and meta["probe"]["vector_fp32_probe_passed"] is True
            and meta["probe"]["mnn_rvv_capability_detected"] is True
            and meta["vlen_expectation_check"] == "passed", "VLEN/probe mismatch: " + run)
    require(meta["changed_source_files"] == [], "Sources changed during run")
    require(len(meta["affinity_effective"]) == 1
            and meta["affinity_requested"] == meta["affinity_effective"], "CPU affinity mismatch")
    validate_build(meta, evidence, run)
    records = [json.loads(line) for line in evidence.text(run + "/results.jsonl").splitlines() if line.strip()]
    require(records and records[0]["record"] == "run" and records[-1]["record"] == "summary",
            "Missing run/terminal summary: " + run)
    require(sum(row["record"] == "run" for row in records) == 1
            and sum(row["record"] == "summary" for row in records) == 1, "Duplicate headers/summaries")
    header, summary = records[0], records[-1]
    correctness_only = run == "correctness"
    require(header["evidence_tier"] == meta["evidence_tier"] and header["seed"] == meta["seed"]
            and header["vlen_bits"] == vlen and header["correctness_only"] is correctness_only,
            "Run header mismatch")
    require(summary == meta["summary"] and summary["failed"] == 0
            and summary["correctness_only"] is correctness_only and meta["correctness_only"] is correctness_only,
            "Summary/metadata mismatch or correctness failures: " + run)
    cases, samples, per_case = {}, {}, {}
    for row in records[1:-1]:
        kind = row["record"]
        key = case_key(row)
        case_id = row["case_id"]
        if kind == "correctness":
            require(case_id not in cases and row["passed"] is True
                    and row["variants"] == 2 and row["datasets"] >= 2, "Failed/duplicate correctness case")
            cases[case_id] = key
        elif kind == "sample":
            variant, round_index = row["variant"], row["round"]
            require(variant in VARIANTS and type(round_index) is int
                    and 0 <= round_index < meta["rounds"], "Invalid variant/round")
            sample_key = (case_id, variant, round_index)
            require(sample_key not in samples, "Duplicate sample: " + str(sample_key))
            require(cases.get(case_id) == key and row["single_thread"] is True,
                    "Sample lacks matching correctness/single-thread check")
            require(type(row["iterations"]) is int and row["iterations"] > 0
                    and type(row["elapsed_ns"]) is int and row["elapsed_ns"] > 0,
                    "Invalid sample duration/iterations")
            number_equal(row["ns_per_call"], row["elapsed_ns"] / row["iterations"], "elapsed/iterations")
            expected_order = "AB" if round_index % 2 == 0 else "BA"
            expected_position = VARIANTS.index(variant) if expected_order == "AB" else 1 - VARIANTS.index(variant)
            require(row["order"] == expected_order and row["position"] == expected_position,
                    "AB/BA order mismatch")
            samples[sample_key] = row
            per_case.setdefault(case_id, []).append(row)
        else:
            raise ValueError("Unexpected record type: " + kind)
    require(len(cases) == summary["cases"] > 0 and len(samples) == summary["samples"], "Record count mismatch")
    require(set(cases) == set(range(len(cases))), "Case ids are not contiguous")
    require(len({key[1:] for key in cases.values()}) == len(cases), "Duplicate case dimensions")
    if correctness_only:
        require(not samples, "Correctness-only run has timing data")
        return meta, cases, {}
    rounds = meta["rounds"]
    require(type(rounds) is int and rounds >= 3, "Too few measurement rounds")
    require(len(samples) == len(cases) * len(VARIANTS) * rounds, "Missing sample rounds")
    csv_samples = evidence.csv(run + "/samples.csv")
    require(len(csv_samples) == len(samples), "samples.csv count mismatch")
    seen = set()
    for row in csv_samples:
        key = (int(row["case_id"]), row["variant"], int(row["round"]))
        require(key in samples and key not in seen, "samples.csv missing/duplicate sample")
        seen.add(key)
        compare_csv(row, samples[key], run + "/samples.csv/" + str(key))
    computed = {}
    for case_id, key in cases.items():
        rows = per_case[case_id]
        timing_inputs = {row["timing_input"] for row in rows}
        require(len(timing_inputs) == 1, "Timing inputs differ within case")
        record = dict(zip(KEY_FIELDS, key))
        record["timing_input"] = timing_inputs.pop()
        for variant, label in zip(VARIANTS, ("scalar", "rvv")):
            values = [samples[(case_id, variant, index)]["ns_per_call"] for index in range(rounds)]
            median = statistics.median(values)
            record.update({label + "_samples": len(values), label + "_median_ns": median,
                           label + "_min_ns": min(values), label + "_max_ns": max(values),
                           label + "_mad_ns": statistics.median(abs(value - median) for value in values)})
        meaningful = record["area"] > 0 and record["channels"] > 0
        record["speedup_scalar_over_rvv"] = record["scalar_median_ns"] / record["rvv_median_ns"] if meaningful else None
        computed[case_id] = record
    csv_comparison = evidence.csv(run + "/comparison.csv")
    require(len(csv_comparison) == len(cases), "comparison.csv count mismatch")
    seen = set()
    for row in csv_comparison:
        case_id = int(row["case_id"])
        require(case_id in computed and case_id not in seen, "comparison.csv missing/duplicate case")
        seen.add(case_id)
        compare_csv(row, computed[case_id], run + "/comparison.csv/" + str(case_id))
    return meta, cases, computed


def aggregate(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    result = []
    for key, members in sorted(groups.items()):
        ratios = [row["ratio_process_median"] for row in members]
        record = dict(zip(fields, key))
        record.update({"cases": len(members), "equal_case_geomean_speedup": math.exp(statistics.mean(map(math.log, ratios))),
                       "median_case_speedup": statistics.median(ratios), "min_case_speedup": min(ratios),
                       "max_case_speedup": max(ratios),
                       "all_processes_gain_gt_5pct": sum(row["classification"] == "all_gain_gt_5pct" for row in members),
                       "all_processes_loss_gt_5pct": sum(row["classification"] == "all_loss_gt_5pct" for row in members),
                       "mixed_or_within_5pct": sum(row["classification"] == "mixed_or_within_5pct" for row in members),
                       "median_ratio_process_max_over_min": statistics.median(row["ratio_process_max_over_min"] for row in members),
                       "worst_ratio_process_max_over_min": max(row["ratio_process_max_over_min"] for row in members)})
        result.append(record)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    require(not args.output.exists(), "Output must be a fresh directory")
    evidence = Evidence(args.session)
    summary = evidence.json("summary.json")
    require(summary["status"] == "passed" and summary["exit_code"] == 0
            and evidence.text("exitcode").strip() == "0", "Session failed")
    require(summary["evidence_tier"] == "target_riscv64_execution" and summary["observed_machine"] == "riscv64",
            "Session is not target RISC-V execution")
    require(summary["performance_processes_requested"] == 3, "Expected three performance processes")
    stages = {stage["name"]: stage for stage in summary["stages"]}
    require(len(stages) == len(summary["stages"]), "Duplicate session stage")
    for name, stage in stages.items():
        require(name.replace("-", "").isalnum(), "Invalid stage name")
        require(stage["status"] == "passed" and stage["exit_code"] == 0
                and evidence.text(name + ".exitcode").strip() == "0", "Failed stage: " + name)
    require(set(RUNS).issubset(stages), "Missing correctness/performance stage")
    vlen = summary["expected_vlen_bits"]
    require(type(vlen) is int and vlen > 0 and summary["probe"]["vlen_bits"] == vlen
            and summary["probe"]["vector_fp32_probe_passed"] is True, "Invalid session RVV probe")
    runs = {name: validate_run(evidence, name, vlen) for name in RUNS}
    reference, keys, _ = runs["correctness"]
    cohort_fields = ("source_hashes", "object_hashes", "generated_scalar_sha256", "binary_sha256", "commands",
                     "invoked_executable_sha256", "compiler_version", "compiler_target", "language_standard",
                     "scalar_extra_flags", "affinity_effective", "seed", "rounds", "sample_ms",
                     "machine", "kernel_release", "expected_vlen_bits", "probe", "evidence_tier")
    for name, (meta, current_keys, _) in runs.items():
        require(current_keys == keys, "Case matrix differs across processes: " + name)
        for field in cohort_fields:
            require(meta[field] == reference[field], "Mixed source/build/experiment cohort: " + field)
        require(meta["affinity_effective"] == summary["affinity_selected"], "Session/run affinity mismatch")
    rows = []
    excluded = []
    for case_id, key in sorted(keys.items()):
        record = dict(zip(KEY_FIELDS, key))
        if not record["area"] or not record["channels"]:
            excluded.append(case_id)
            continue
        cases = [runs[name][2][case_id] for name in RUNS[1:]]
        require(len({case["timing_input"] for case in cases}) == 1, "Timing input differs across processes")
        record["timing_input"] = cases[0]["timing_input"]
        ratios = [case["speedup_scalar_over_rvv"] for case in cases]
        record["ratio_process_median"] = statistics.median(ratios)
        record["ratio_process_min"] = min(ratios)
        record["ratio_process_max"] = max(ratios)
        record["ratio_process_max_over_min"] = max(ratios) / min(ratios)
        record["classification"] = ("all_gain_gt_5pct" if min(ratios) > 1.05 else
                                    "all_loss_gt_5pct" if max(ratios) < 0.95 else "mixed_or_within_5pct")
        for index, case in enumerate(cases, 1):
            record["ratio_p" + str(index)] = case["speedup_scalar_over_rvv"]
            for label in ("scalar", "rvv"):
                median = case[label + "_median_ns"]
                record[label + "_median_ns_p" + str(index)] = median
                record[label + "_mad_pct_p" + str(index)] = 100 * case[label + "_mad_ns"] / median
                record[label + "_sample_max_over_min_p" + str(index)] = case[label + "_max_ns"] / case[label + "_min_ns"]
        for label in ("scalar", "rvv"):
            times = [case[label + "_median_ns"] for case in cases]
            record[label + "_process_median_ns"] = statistics.median(times)
            record[label + "_process_max_over_min"] = max(times) / min(times)
        rows.append(record)
    require(rows, "No nonzero-work cases")
    contexts = {name: evidence.json(name + ".context.json") for name in
                ("preflight",) + tuple(run + suffix for run in RUNS[1:] for suffix in ("-before", "-after"))}
    for context in contexts.values():
        require(context["machine"] == "riscv64" and context["online_cpu_count"] > 0
                and context["kernel_release"] == reference["kernel_release"]
                and context["selected_cpus"] == reference["affinity_effective"],
                "Invalid hardware context")
        require(set(context["selected_cpus"]).issubset(context["allowed_cpus"]), "Selected CPU was not allowed")
        require(context["cpu_information"] == contexts["preflight"]["cpu_information"],
                "CPU identity changed between context snapshots")
    cpu_identity = {}
    for line in contexts["preflight"]["cpu_information"]:
        if ":" in line:
            name, value = (part.strip() for part in line.split(":", 1))
            if name in ("isa", "mvendorid", "marchid", "mimpid", "mmu", "uarch", "model name"):
                cpu_identity.setdefault(name, set()).add(value)
    load = [context["load_average"][0] for context in contexts.values()]
    counts = sorted({context["online_cpu_count"] for context in contexts.values()})
    limits = ["仅适用于本次目标及参数矩阵；硬件型号和非模拟环境仍需外部确认。",
              "直接函数调用微基准；不证明完整 MNN 派发、模型速度或多核扩展。",
              "几何均值对非零配置等权，是合成矩阵汇总，不能解释为模型加速比。",
              "三进程全部 >1.05 / <0.95 是描述性筛选，未进行统计显著性检验。",
              "温度/频率仅为运行前后快照；没有持续监控或 inert 对照。"]
    if any(context["load_average"][0] > context["online_cpu_count"] for context in contexts.values()):
        limits.append("系统 1 分钟负载高于在线 CPU 数；绑核不等于独占，需低负载复测收益及阈值。")
    if any(not context.get("thermal") for context in contexts.values()):
        limits.append("温度读数缺失，无法排除热漂移。")
    if summary["integration"] == "explicitly_skipped":
        limits.append("完整 MNN 集成测试已显式跳过。")
    archive = None
    if args.archive:
        checksum = args.checksum or Path(str(args.archive) + ".sha256")
        tokens = checksum.read_text(encoding="utf-8").strip().split()
        require(tokens and len(tokens[0]) == 64, "Invalid archive checksum file")
        hasher = hashlib.sha256()
        with args.archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        require(hasher.hexdigest() == tokens[0].lower(), "Archive checksum mismatch")
        archive = {"name": args.archive.name, "sha256": hasher.hexdigest(), "checksum_verified": True,
                   "extraction_relationship": "Archive bytes verified; caller is responsible for safe extraction into SESSION"}
    require(args.archive or not args.checksum, "--checksum requires --archive")
    by_function, by_area = aggregate(rows, ("op",)), aggregate(rows, ("op", "area"))
    analysis = {
        "schema_version": 1, "status": "validated", "session": args.session.name,
        "evidence_tier": "target_riscv64_execution", "archive": archive,
        "unique_cases": len(keys), "excluded_zero_work_cases": len(excluded), "excluded_case_ids": excluded,
        "analyzed_cases": len(rows), "performance_processes": 3, "rounds_per_variant_per_process": reference["rounds"],
        "sample_records": sum(runs[name][0]["summary"]["samples"] for name in RUNS[1:]),
        "correctness_runs": {name: runs[name][0]["summary"] for name in RUNS},
        "method": {"per_process": "median scalar ns/call divided by median RVV ns/call, recomputed from JSONL",
                   "cross_process": "median of the three per-process ratios, not ratio of pooled timing medians",
                   "mad": "median absolute deviation from each process/variant median; percent = 100 * MAD / median",
                   "classification": "all three ratios > 1.05, all three ratios < 0.95, or mixed/within 5%; descriptive only",
                   "geomean": "exp(mean(log(case ratio process median))); equally weighted synthetic nonzero case matrix",
                   "numeric_tolerance": "rel_tol=1e-10, abs_tol=1e-9, permits printed JSON floating-point rounding"},
        "cohort_verified": {field: reference[field] for field in cohort_fields},
        "environment": {"vlen_bits": vlen, "cpu_affinity": reference["affinity_effective"], "online_cpu_counts": counts,
                        "cpu_identity_fields": {name: sorted(values) for name, values in cpu_identity.items()},
                        "load_1min_min": min(load), "load_1min_max": max(load),
                        "frequency_snapshots": {name: context.get("frequency", {}) for name, context in contexts.items()},
                        "thermal_snapshots": {name: context.get("thermal", {}) for name, context in contexts.items()}},
        "integration": summary["integration"], "limitations": limits, "by_function": by_function,
        "raw_evidence_sha256": dict(sorted(evidence.hashes.items())),
    }
    lines = ["# RVV 与原始标量函数：结果核验与性能分析", "",
             "原始 JSONL、samples.csv、comparison.csv 逐项核验通过；源码、对象文件和二进制属于同一批次。",
             "4 次运行各有 {} 个正确性配置，失败为 0；3 个性能进程共 {} 条采样。".format(len(keys), analysis["sample_records"]),
             "排除 {} 个零工作量配置，分析 {} 个非零配置。".format(len(excluded), len(rows)), "",
             "| 函数 | 配置数 | 等权几何均值 | 配置中位数 | 三次均 >1.05 | 三次均 <0.95 | 其余 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in by_function:
        lines.append("| {op} | {cases} | {equal_case_geomean_speedup:.3f}× | {median_case_speedup:.3f}× | "
                     "{all_processes_gain_gt_5pct} | {all_processes_loss_gt_5pct} | {mixed_or_within_5pct} |".format(**row))
    lines += ["", "每个配置先分别计算三次进程内标量/RVV 耗时中位数之比，再取这三个比值的中位数。",
              "cases.csv 保留各进程比值、耗时中位数、MAD 百分比及最大/最小离散比；by_area.csv 按函数与 area 分组。",
              "", "目标 VLEN={} bits，绑核 {}，在线 CPU 数 {}；1 分钟负载范围 {:.2f}–{:.2f}。".format(
                  vlen, reference["affinity_effective"], counts, min(load), max(load)), ""]
    lines.extend("- " + limitation for limitation in limits)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "cases.csv", rows)
    write_csv(args.output / "by_function.csv", by_function)
    write_csv(args.output / "by_area.csv", by_area)
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "validated", "output": str(args.output.resolve()), "analyzed_cases": len(rows),
            "excluded_zero_work_cases": len(excluded), "by_function": by_function}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Already safely extracted session directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; never overwrites")
    parser.add_argument("--archive", type=Path, help="Optional original archive to hash; never extracts or executes it")
    parser.add_argument("--checksum", type=Path, help="Archive checksum (default: archive path plus .sha256)")
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args), ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        print("FAILED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

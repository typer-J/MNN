#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:-"-O3 -std=c++17 -march=rv64gcv -mabi=lp64d -fno-exceptions -fno-rtti"}
EVENTS=${EVENTS:-"duration_time,cycles,instructions,cache-references,cache-misses,branches,branch-misses"}
REPEAT=${REPEAT:-1}
TASKSET_CPU=${TASKSET_CPU:-}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
mkdir -p "${BUILD_DIR}"

targets=(
  MNNAccumulateSequenceNumber
  MNNSumByAxisLForMatmul_A
  MNNReorderWeightInt4
  MNNSumWeightInt8
  MNNPackedMatMul_int8
  MNNPackedMatMulRemain_int8
  MNNAbsMaxFP32
  MNNDynamicQuantFP32
  MNNAsyQuantFunc
  MNNAsyQuantInfo_FP32
)

compile_one() {
  local name="$1"
  local src="${ROOT_DIR}/test_perf_${name}.cpp"
  local bin="${BUILD_DIR}/test_perf_${name}_bin"
  echo "[BUILD] ${name}"
  "${CXX}" ${CXXFLAGS} "${src}" -o "${bin}"
}

run_perf() {
  local name="$1"
  local mode="$2"
  local bin="${BUILD_DIR}/test_perf_${name}_bin"
  local cmd=(perf stat -r "${REPEAT}" -e "${EVENTS}" -- "${bin}" "${mode}")
  if [[ -n "${TASKSET_CPU}" ]]; then
    cmd=(taskset -c "${TASKSET_CPU}" "${cmd[@]}")
  fi
  "${cmd[@]}"
}

if [[ "${1:-}" == "--build-only" ]]; then
  for name in "${targets[@]}"; do
    compile_one "${name}"
  done
  exit 0
fi

if [[ $# -gt 0 ]]; then
  targets=("$@")
fi

for name in "${targets[@]}"; do
  compile_one "${name}"
  echo "=============================================================="
  echo "PROFILING TARGET: [ ${name} ]"
  echo "=============================================================="
  echo ">>> [1/2] Scalar Reference Mode ..."
  run_perf "${name}" 0
  echo
  echo ">>> [2/2] RVV Optimized Mode ..."
  run_perf "${name}" 1
  echo "--------------------------------------------------------------"
  echo
done

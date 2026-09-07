#!/usr/bin/env bash
# Run from an isolated checkout; every generated file stays in a fresh result directory.
# Usage: ORIGINAL_REPO=<protected-checkout> CC=/path/gcc CXX=/path/g++ bash integration.sh <fresh-output-dir>
set -u -o pipefail

if [[ $# != 1 ]]; then
    printf 'Usage: ORIGINAL_REPO=<protected-checkout> CC=<GCC14> CXX=<GXX14> %s <fresh-output-dir>\n' "$0" >&2
    exit 2
fi
if [[ $(uname -s) != Linux || $(uname -m) != riscv64 ]]; then
    printf 'Refusing integration: native Linux riscv64 is required; no output was created.\n' >&2
    exit 2
fi
PYTHON_BIN=${PYTHON:-python3}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { printf 'Python 3 is required.\n' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
SOURCE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P) || exit 2
OUT=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1") || exit 2
if [[ -z $1 || $OUT == *$'\n'* || $OUT == *$'\t'* || -e $OUT || -L $1 ]]; then
    printf 'Output must be a fresh, nonexistent directory without tabs/newlines: %s\n' "$1" >&2
    exit 2
fi
"$PYTHON_BIN" - "$SOURCE_ROOT" "$OUT" <<'PY' || exit 2
import os, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:])
original_value = os.environ.get('ORIGINAL_REPO')
if not original_value:
    sys.exit('ORIGINAL_REPO is required; no output was created.')
original = pathlib.Path(original_value).resolve()
def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
if not original.is_dir() or within(root, original) or within(out, original) or not within(out, root) or out == root:
    sys.exit('Candidate must be outside ORIGINAL_REPO and output inside the candidate; no output was created.')
PY
mkdir -p -- "$(dirname -- "$OUT")" && mkdir -- "$OUT" || exit 2
mkdir -- "$OUT/logs" "$OUT/helpers" "$OUT/evidence" || exit 2
printf 'step\texit_code\tlog\tcommand\n' > "$OUT/steps.tsv"
JOBS=${JOBS:-4}
failure_reason='Run did not complete.'
op_status=not_run
dispatch_status=not_verified
dispatch_reason='GDB was not available or the run stopped before dispatch validation.'

finish() {
    local rc=$1
    trap - EXIT
    "$PYTHON_BIN" - "$OUT" "$rc" "$failure_reason" "$op_status" "$dispatch_status" "$dispatch_reason" <<'PY'
import csv, json, pathlib, sys
out, rc, reason, op_status, dispatch_status, dispatch_reason = sys.argv[1:]
out = pathlib.Path(out)
with (out / 'steps.tsv').open() as stream:
    steps = list(csv.DictReader(stream, delimiter='\t'))
for step in steps:
    step['exit_code'] = int(step['exit_code'])
results = []
for path in sorted((out / 'evidence').glob('*.result.json')):
    results.append(json.loads(path.read_text()))
status = 'failed' if int(rc) else ('passed' if dispatch_status == 'verified' else 'op_tests_passed_dispatch_' + dispatch_status)
summary = dict(status=status, exit_code=int(rc), failure_reason=reason if int(rc) else None,
               execution_tier='target_riscv64_execution', op_tests_status=op_status,
               dispatch_status=dispatch_status, dispatch_reason=dispatch_reason,
               model_stage='not_run', model_reason='No model is downloaded or selected by this integration runner.',
               performance_stage='not_run', results=results, steps=steps)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Integration summary:', out / 'summary.json')
PY
    local summary_rc=$?
    if (( summary_rc != 0 && rc == 0 )); then rc=$summary_rc; fi
    exit "$rc"
}
trap 'finish "$?"' EXIT
trap 'failure_reason="Interrupted by signal."; exit 130' INT
trap 'failure_reason="Terminated by signal."; exit 143' TERM
fail() { failure_reason=$1; printf '%s\n' "$1" >&2; exit "${2:-1}"; }

run_step() {
    local name=$1 rc
    shift
    {
        printf '#!/usr/bin/env bash\n'
        printf 'cd -- %q\n' "$OUT"
        printf '%q ' "$@"
        printf '\n'
    } > "$OUT/logs/$name.command.sh"
    printf '[integration] %s\n' "$name"
    (cd -- "$OUT" && "$@") > "$OUT/logs/$name.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$OUT/logs/$name.exitcode"
    printf '%s\t%s\tlogs/%s.log\tlogs/%s.command.sh\n' "$name" "$rc" "$name" "$name" >> "$OUT/steps.tsv"
    return "$rc"
}

[[ $JOBS =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be a positive integer.' 2
[[ -f $SOURCE_ROOT/CMakeLists.txt && -f $SOURCE_ROOT/test/main.cpp ]] || fail 'Cannot locate the MNN source checkout.' 2
[[ -n ${CC:-} && -n ${CXX:-} ]] || fail 'Set CC and CXX to GCC 14 compiler executables (without arguments).' 2
CC_BIN=$(command -v "$CC") || fail 'CC is not an executable command.' 2
CXX_BIN=$(command -v "$CXX") || fail 'CXX is not an executable command.' 2
CC_BIN=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CC_BIN") || fail 'Cannot resolve CC.' 2
CXX_BIN=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CXX_BIN") || fail 'Cannot resolve CXX.' 2
command -v cmake >/dev/null 2>&1 || fail 'CMake is required; dependencies are never installed automatically.' 2
command -v ldd >/dev/null 2>&1 || fail 'An existing ldd is required to record runtime library resolution.' 2
PIN=()
if [[ -n ${CPUSET:-} ]]; then
    [[ $CPUSET =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'CPUSET must be an explicit comma-separated list of CPU numbers.' 2
    command -v taskset >/dev/null 2>&1 || fail 'CPUSET requires an existing taskset executable.' 2
    PIN=(taskset -c "$CPUSET")
fi
for kind in cc cxx; do
    compiler=$CC_BIN
    [[ $kind == cxx ]] && compiler=$CXX_BIN
    run_step "$kind-version" "$compiler" --version || fail "$kind --version failed."
    run_step "$kind-dumpversion" "$compiler" -dumpfullversion || fail "$kind version query failed."
    run_step "$kind-dumpmachine" "$compiler" -dumpmachine || fail "$kind target query failed."
    version=$(<"$OUT/logs/$kind-dumpversion.log")
    machine=$(<"$OUT/logs/$kind-dumpmachine.log")
    [[ $version == 14.* && $machine == riscv64* ]] || fail "$kind must be native RISC-V GCC 14; found version=$version target=$machine."
done
run_step cmake-version cmake --version || fail 'CMake version query failed.'

# Parse the actual MNN report prefix, not TEST_CASE= (a duplicate summary).
cat > "$OUT/helpers/check_results.py" <<'PY'
import json, pathlib, re, sys
log, dest, name, expected_reports, process_rc = sys.argv[1:]
text = pathlib.Path(log).read_text(errors='replace')
reports, errors = [], []
for line in text.splitlines():
    match = re.match(r'^TEST_CASE_AMOUNT_UNIT[^:]*:\s*(\{.*\})\s*$', line.strip())
    if match:
        try:
            reports.append(json.loads(match.group(1)))
        except ValueError as exc:
            errors.append('Invalid summary JSON: ' + str(exc))
if int(process_rc):
    errors.append('Process exited with code ' + process_rc)
if len(reports) != int(expected_reports):
    errors.append('Expected %s summaries, got %s' % (expected_reports, len(reports)))
for report in reports:
    if any(type(report.get(key)) is not int for key in ('passed', 'failed', 'blocked', 'skipped')):
        errors.append('Missing or non-integer summary fields')
    elif report['passed'] <= 0 or report['failed'] != 0 or report['blocked'] != 0 or report['skipped'] != 0:
        errors.append('Empty, failed, blocked or skipped test summary')
if name == 'dispatch':
    match = re.search(r'RVV_DISPATCH pack=(\d+) unpack=(\d+) scale=(\d+) relu=(\d+)', text)
    hits = dict(zip(('pack', 'unpack', 'scale', 'relu'), map(int, match.groups()))) if match else {}
    if not hits or min(hits.values()) < 1:
        errors.append('Missing one or more RVV dispatch hits')
    required_names = ('op/scale', 'op/prelu')
else:
    hits = None
    required_names = (name,)
for test_name in required_names:
    if not re.search(r'\brunning ' + re.escape(test_name) + r'\.', text):
        errors.append('Missing exact running test name: ' + test_name)
result = dict(name=name, process_exit_code=int(process_rc), reports=reports, dispatch_hits=hits,
              status='failed' if errors else 'passed', errors=errors)
pathlib.Path(dest).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
sys.exit(bool(errors))
PY

cat > "$OUT/helpers/check_build.py" <<'PY'
import json, pathlib, shlex, sys
build, variant, dest = sys.argv[1:]
entries = json.loads((pathlib.Path(build) / 'compile_commands.json').read_text())
errors, rvv_count = [], 0
for entry in entries:
    args = entry.get('arguments') or shlex.split(entry['command'])
    if not entry['file'].endswith(('.c', '.cpp', '.cc', '.cxx')):
        continue
    for flag in ('-fno-tree-vectorize', '-fno-tree-slp-vectorize', '-fno-fast-math', '-ffp-contract=off'):
        if flag not in args:
            errors.append('Required global flag absent: ' + flag)
    march = [arg.split('=', 1)[1] for arg in args if arg.startswith('-march=')]
    is_rvv = '/source/backend/cpu/riscv/rvv/' in entry['file'].replace('\\', '/')
    rvv_count += is_rvv
    expected = 'rv64gcv' if variant == 'rvv' and is_rvv else 'rv64gc'
    if not march or march[-1] != expected:
        errors.append('Unexpected final -march: ' + entry['file'])
    if any('xsmtvdotii' in arg or arg == '-ffast-math' for arg in args):
        errors.append('Vendor ISA or fast-math present')
if (variant == 'rvv' and not rvv_count) or (variant == 'scalar' and rvv_count):
    errors.append('Unexpected RVV translation-unit count')
result = dict(variant=variant, translation_units=len(entries), rvv_translation_units=rvv_count,
              required_generic_isa='rv64gc', autovectorization=False, errors=sorted(set(errors)))
pathlib.Path(dest).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
sys.exit(bool(errors))
PY

run_step metadata "${PIN[@]}" "$PYTHON_BIN" -c '
import hashlib,json,os,pathlib,platform,subprocess,sys
root,out,cc,cxx=sys.argv[1:]; root=pathlib.Path(root); out=pathlib.Path(out)
scope=["benchmark/rvv_scalar_compare/integration.sh","source/backend/cpu/compute/CommonOptFunction.cpp",
       "source/backend/cpu/compute/CommonOptFunction.h"]
scope += ["source/backend/cpu/riscv/rvv/"+n+".cpp" for n in
          ("MNNPackC4","MNNUnpackC4","MNNScaleAndAddBias","MNNReluWithSlopeChannel")]
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in scope}
try:
    git=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True)
    revision=git.stdout.strip() if git.returncode==0 else None
except FileNotFoundError: revision=None
meta=dict(source_revision=revision,source_hashes=hashes,compiler_c=cc,compiler_cxx=cxx,
          system=platform.platform(),requested_cpuset=os.getenv("CPUSET"),expected_vlen=os.getenv("EXPECT_VLEN"),
          measured_vlen=None,affinity=sorted(os.sched_getaffinity(0)),model_stage="not_run")
(out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta))
' "$SOURCE_ROOT" "$OUT" "$CC_BIN" "$CXX_BIN" || fail 'Metadata collection failed.'

COMMON_FLAGS='-march=rv64gc -mabi=lp64d -fno-tree-vectorize -fno-tree-slp-vectorize -fno-fast-math -ffp-contract=off -g'
for variant in scalar rvv; do
    build=$OUT/build-$variant
    rvv=OFF
    [[ $variant == rvv ]] && rvv=ON
    run_step "$variant-configure" cmake -S "$SOURCE_ROOT" -B "$build" \
        -DCMAKE_C_COMPILER="$CC_BIN" -DCMAKE_CXX_COMPILER="$CXX_BIN" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_FLAGS="$COMMON_FLAGS" -DCMAKE_CXX_FLAGS="$COMMON_FLAGS" \
        -DMNN_USE_RVV="$rvv" -DMNN_RVV_MARCH=rv64gcv \
        -DMNN_RVV_SPACEMIT_IME2=OFF -DMNN_RVV_FAST_MATH=OFF \
        -DMNN_BUILD_TEST=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
        -DMNN_BUILD_LLM=ON -DMNN_USE_THREAD_POOL=ON -DMNN_OPENMP=OFF \
        -DMNN_BUILD_SHARED_LIBS=ON -DMNN_SEP_BUILD=OFF \
        || fail "$variant configure failed; see logs."
    run_step "$variant-build-flags" "$PYTHON_BIN" "$OUT/helpers/check_build.py" "$build" "$variant" \
        "$OUT/evidence/$variant-build-flags.json" || fail "$variant build flags do not match the requested baseline."
    run_step "$variant-build" cmake --build "$build" --parallel "$JOBS" || fail "$variant full MNN build failed; see logs."
    binary=$build/run_test.out
    [[ -x $binary ]] || fail "Missing expected test binary: $binary"
    run_step "$variant-artifacts" "$PYTHON_BIN" -c '
import hashlib,json,pathlib,sys
build,dest=map(pathlib.Path,sys.argv[1:]); files=[build/"run_test.out"]+sorted(build.glob("libMNN*.so*"))
records=[]
for path in files:
    if not path.is_file(): continue
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    records.append(dict(name=path.name,resolved=str(path.resolve()),size=path.stat().st_size,sha256=digest.hexdigest()))
if not any(record["name"].startswith("libMNN") for record in records): raise SystemExit("No built libMNN shared library found")
dest.write_text(json.dumps(records,indent=2)+"\n"); print(json.dumps(records))
' "$build" "$OUT/evidence/$variant-artifacts.json" || fail "$variant artifact identity collection failed."
    run_step "$variant-ldd" env "LD_LIBRARY_PATH=$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$binary" \
        || fail "$variant runtime library resolution failed."
    # Keep tests and any emitted test files in OUT, never in the source checkout.
    for threads in 1 4; do
        for test_name in op/convert op/scale op/prelu op/relu; do
            name=$variant-${test_name//\//-}-t$threads
            run_step "$name" env -u MNN_TEST_SKIP -u MNN_CPU_USE_DEFAULT_BACKEND -u MNN_CPU_TARGET \
                "LD_LIBRARY_PATH=$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                "${PIN[@]}" "$binary" "$test_name" 0 1 "$threads" sg2044_integration 0
            test_rc=$?
            run_step "$name-check" "$PYTHON_BIN" "$OUT/helpers/check_results.py" "$OUT/logs/$name.log" \
                "$OUT/evidence/$name.result.json" "$test_name" 1 "$test_rc" \
                || fail "$name failed, was empty, or emitted an invalid summary."
        done
    done
done
op_status=passed

if command -v gdb >/dev/null 2>&1; then
    gdb_script=$OUT/helpers/dispatch.gdb
    {
        printf 'set pagination off\nset confirm off\nset breakpoint pending on\n'
        for counter in pack unpack scale relu; do printf 'set $%s = 0\n' "$counter"; done
        symbols=(MNNPackC4_RVV MNNUnpackC4_RVV MNNScaleAndAddBias_RVV MNNReluWithSlopeChannel_RVV)
        counters=(pack unpack scale relu)
        for i in 0 1 2 3; do
            printf 'break %s\ncommands\nsilent\n' "${symbols[$i]}"
            printf 'set $%s = $%s + 1\nif $%s == 1\nbt 8\nend\ncontinue\nend\n' "${counters[$i]}" "${counters[$i]}" "${counters[$i]}"
        done
        for test_name in op/scale op/prelu; do
            printf 'run %s 0 1 1 sg2044_dispatch 0\nif $_exitcode != 0\nquit 1\nend\n' "$test_name"
        done
        printf 'printf "RVV_DISPATCH pack=%%d unpack=%%d scale=%%d relu=%%d\\n", $pack, $unpack, $scale, $relu\n'
        printf 'if $pack < 1 || $unpack < 1 || $scale < 1 || $relu < 1\nquit 1\nend\nquit 0\n'
    } > "$gdb_script"
    run_step dispatch-gdb env -u MNN_TEST_SKIP -u MNN_CPU_USE_DEFAULT_BACKEND -u MNN_CPU_TARGET \
        "LD_LIBRARY_PATH=$OUT/build-rvv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "${PIN[@]}" gdb -q -nx --batch -x "$gdb_script" --args "$OUT/build-rvv/run_test.out"
    gdb_rc=$?
    run_step dispatch-check "$PYTHON_BIN" "$OUT/helpers/check_results.py" "$OUT/logs/dispatch-gdb.log" \
        "$OUT/evidence/dispatch.result.json" dispatch 2 "$gdb_rc"
    check_rc=$?
    if (( check_rc == 0 )); then
        dispatch_status=verified
        dispatch_reason='All four RVV symbols were hit by passing MNN scale/prelu tests under GDB.'
    elif "$PYTHON_BIN" -c '
import pathlib,re,sys
t=pathlib.Path(sys.argv[1]).read_text(errors="replace")
sys.exit(0 if re.search(r"(?:ptrace.*(?:not permitted|denied)|Could not trace|Could not attach|vfork: Operation not permitted)",t,re.I) else 1)
' "$OUT/logs/dispatch-gdb.log"; then
        dispatch_status=blocked
        dispatch_reason='GDB/ptrace was blocked by the environment; ordinary op results remain valid.'
    else
        dispatch_status=failed
        dispatch_reason='GDB execution, nonempty op results, or one of the four required symbol hits failed.'
        fail "$dispatch_reason"
    fi
fi
failure_reason=''
exit 0

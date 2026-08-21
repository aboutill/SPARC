#!/usr/bin/env bash
# =============================================================================
# tests.bash: part of sparc package.
#
# Internal script to test pipeline behaviour across a handful of
# representative configurations.
# =============================================================================

set -euo pipefail

readonly HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
readonly TEST_DIR="${HOST_DIR}/tests"
readonly CFG_DIR="${TEST_DIR}/cfg"
readonly CONF_DIR="${TEST_DIR}/conf"
readonly MODELS_DIR="${TEST_DIR}/models"
readonly INPUT_DIR="${TEST_DIR}/in"
readonly OUTPUT_DIR="${TEST_DIR}/out"
readonly SUB_ID="test001"

command -v SPARC >/dev/null 2>&1 || { echo "Error: SPARC not found on PATH." >&2; exit 1; }
[[ -d "${INPUT_DIR}" ]] || { echo "Error: ${INPUT_DIR} not found." >&2; exit 1; }
[[ -d "${CFG_DIR}" ]] || { echo "Error: ${CFG_DIR} not found." >&2; exit 1; }
[[ -d "${MODELS_DIR}" ]] || { echo "Error: ${MODELS_DIR} not found." >&2; exit 1; }

# Clean up previous runs
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Test runner
# =============================================================================
TOTAL=0
FAILURES=0

run_test() {
  local description="$1"
  shift
  TOTAL=$((TOTAL + 1))
  echo "=== ${description} ==="
  if SPARC pipeline "$@"; then
    echo "--- PASS: ${description} ---"
  else
    echo "--- FAIL: ${description} ---" >&2
    FAILURES=$((FAILURES + 1))
  fi
  echo
}

# =============================================================================
# Tests
# =============================================================================
run_test "Test 1: --conf" \
  --conf="${CONF_DIR}/pipeline.conf" \
  --mode="full_auto"

run_test "Test 2: minimal required args" \
  --input="${INPUT_DIR}/${SUB_ID}" \
  --output="${OUTPUT_DIR}/${SUB_ID}_002" \
  --mode="full_auto"

run_test "Test 3: debug/profile/log/verbose flags" \
  --input="${INPUT_DIR}/${SUB_ID}" \
  --output="${OUTPUT_DIR}/${SUB_ID}_003" \
  --mode="full_auto" \
  --file_prefix="${SUB_ID}" \
  --debug \
  --profile \
  --log \
  --verbose

run_test "Test 4: custom models for all three components" \
  --input="${INPUT_DIR}/${SUB_ID}" \
  --output="${OUTPUT_DIR}/${SUB_ID}_004" \
  --mode="full_auto" \
  --file_prefix="${SUB_ID}" \
  --models_chest_seg="${MODELS_DIR}/chest_seg" \
  --models_chest_seg_cfg="${CFG_DIR}/chest_seg.yaml" \
  --models_heart_seg="${MODELS_DIR}/heart_seg" \
  --models_heart_seg_cfg="${CFG_DIR}/heart_seg.yaml" \
  --models_reo="${MODELS_DIR}/reo" \
  --models_reo_cfg="${CFG_DIR}/reo.yaml"

modes=("manual" "semi_auto" "monitored_auto" "full_auto")
for mode in "${modes[@]}"; do
  run_test "Test 5: mode=${mode}" \
    --input="${INPUT_DIR}/${SUB_ID}" \
    --output="${OUTPUT_DIR}/${SUB_ID}_005_${mode}" \
    --file_prefix="${SUB_ID}" \
    --mode="${mode}"
done

gui_modes=("docker" "native")
for gui_mode in "${gui_modes[@]}"; do
  run_test "Test 6: gui_mode=${gui_mode}" \
    --input="${INPUT_DIR}/${SUB_ID}" \
    --output="${OUTPUT_DIR}/${SUB_ID}_006_${gui_mode}" \
    --file_prefix="${SUB_ID}" \
    --mode="semi_auto" \
    --gui_mode="${gui_mode}"
done

run_test "Test 7: manual stack review" \
  --input="${INPUT_DIR}/${SUB_ID}" \
  --output="${OUTPUT_DIR}/${SUB_ID}_007" \
  --file_prefix="${SUB_ID}" \
  --mode="semi_auto" \
  --manual_stack_review
  
cfgs=("pipeline1" "pipeline2")
for cfg in "${cfgs[@]}"; do
  run_test "Test 8: cfg=${CFG_DIR}/${cfg}.yaml" \
    --input="${INPUT_DIR}/${SUB_ID}" \
    --output="${OUTPUT_DIR}/${SUB_ID}_008_${cfg}" \
    --file_prefix="${SUB_ID}" \
    --mode="full_auto" \
    --cfg="${CFG_DIR}/${cfg}.yaml"
done

run_test "Test 9: batch processings" \
  --input="${INPUT_DIR}" \
  --output="${OUTPUT_DIR}/batch" \
  --mode="full_auto" \
  --batch
  
# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "${TOTAL} test(s) run, $((TOTAL - FAILURES)) passed, ${FAILURES} failed."
echo "=============================================="
[[ "${FAILURES}" -eq 0 ]] || exit 1
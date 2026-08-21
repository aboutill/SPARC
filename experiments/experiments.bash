#!/usr/bin/env bash
# =============================================================================
# experiments.bash: part of sparc package.
#
# Script to reproduce paper's experiments.
# =============================================================================

set -euo pipefail

readonly HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
readonly DATA_DIR="${HOST_DIR}/data"
readonly DATA_TRAIN_DIR="${DATA_DIR}/train"
readonly DATA_TEST_DIR="${DATA_DIR}/test"
readonly EXP_DIR="${HOST_DIR}/experiments"
readonly EXP_TRAIN_DIR="${EXP_DIR}/train"
readonly EXP_TEST_DIR="${EXP_DIR}/test"
readonly CFG_DIR="${HOST_DIR}/cfg/models"

readonly SRC_DOMAIN="philips"
readonly TGT_DOMAIN="siemens"
readonly DOMAINS=("${SRC_DOMAIN}" "${TGT_DOMAIN}")

readonly TASKS=("chest_seg" "heart_seg" "reo")
readonly TASKS_CMD=("segmentation" "segmentation" "reorientation")
readonly TASKS_DIR=("seg" "seg" "reo")

command -v SPARC >/dev/null 2>&1 || { echo "Error: SPARC not found on PATH." >&2; exit 1; }
[[ -d "${DATA_TRAIN_DIR}" ]] || { echo "Error: ${DATA_TRAIN_DIR} not found." >&2; exit 1; }
[[ -d "${DATA_TEST_DIR}" ]] || { echo "Error: ${DATA_TEST_DIR} not found." >&2; exit 1; }
[[ -d "${CFG_DIR}" ]] || { echo "Error: ${CFG_DIR} not found." >&2; exit 1; }

# =============================================================================
# Source and target domains trained and evaluated independently
# Source model is also evaluated on target domain
# =============================================================================
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  task_cmd="${TASKS_CMD[$i]}"
  task_dir="${TASKS_DIR[$i]}"

  for domain in "${DOMAINS[@]}"; do
    echo "=== Train: task=${task} domain=${domain} ==="
    train_args=(
      --input="${DATA_TRAIN_DIR}/${task_dir}_${domain}"
      --output="${EXP_TRAIN_DIR}/${task}_${domain}"
      --task="${task_cmd}"
      --cfg="${CFG_DIR}/${task}_${domain}.yaml"
      --log
    )
    SPARC train "${train_args[@]}"
    
    echo "=== Test: task=${task} domain=${domain} on domain=${domain} ==="
    test_args=(
      --input="${DATA_TEST_DIR}/${task_dir}_${domain}"
      --output="${EXP_TEST_DIR}/${task}_${domain}"
      --task="${task_cmd}"
      --cfg="${CFG_DIR}/${task}_${domain}.yaml"
      --models="${EXP_TRAIN_DIR}/${task}_${domain}/models"
      --log
      --save_qc
      --save_indiv
    )
    SPARC test "${test_args[@]}"
  done
  
  echo "=== Test: task=${task} domain=${SRC_DOMAIN} on domain=${TGT_DOMAIN} ==="
  test_args=(
    --input="${DATA_TEST_DIR}/${task_dir}_${TGT_DOMAIN}"
    --output="${EXP_TEST_DIR}/${task}_${SRC_DOMAIN}_to_${TGT_DOMAIN}"
    --task="${task_cmd}"
    --cfg="${CFG_DIR}/${task}_${SRC_DOMAIN}.yaml"
    --models="${EXP_TRAIN_DIR}/${task}_${SRC_DOMAIN}/models"
    --log
    --save_qc
    --save_indiv
  )
  SPARC test "${test_args[@]}"
done

# =============================================================================
# Joint training: both domains trained together and evaluated independently
# =============================================================================
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  task_cmd="${TASKS_CMD[$i]}"
  task_dir="${TASKS_DIR[$i]}"
  echo "=== Train: task=${task} Joint ==="
  train_args=(
    --input="${DATA_TRAIN_DIR}/${task_dir}_${SRC_DOMAIN}"
    --input="${DATA_TRAIN_DIR}/${task_dir}_${TGT_DOMAIN}"
    --output="${EXP_TRAIN_DIR}/${task}_joint"
    --task="${task_cmd}"
    --cfg="${CFG_DIR}/${task}_joint.yaml"
    --log
  )
  SPARC train "${train_args[@]}"
  
  for domain in "${DOMAINS[@]}"; do
    echo "=== Test: task=${task} Joint on domain=${domain} ==="
    test_args=(
      --input="${DATA_TEST_DIR}/${task_dir}_${domain}"
      --output="${EXP_TEST_DIR}/${task}_joint_to_${domain}"
      --task="${task_cmd}"
      --cfg="${CFG_DIR}/${task}_joint.yaml"
      --models="${EXP_TRAIN_DIR}/${task}_joint/models"
      --log
      --save_qc
      --save_indiv
    )
    SPARC test "${test_args[@]}"
  done
done

# =============================================================================
# Transfer learning: fine-tune on target domain from source-domain weights
# =============================================================================
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  task_cmd="${TASKS_CMD[$i]}"
  task_dir="${TASKS_DIR[$i]}"
  echo "=== Train: task=${task} Transfer Learning ==="
  train_args=(
    --input="${DATA_TRAIN_DIR}/${task_dir}_${TGT_DOMAIN}"
    --output="${EXP_TRAIN_DIR}/${task}_${TGT_DOMAIN}_transfer"
    --task="${task_cmd}"
    --cfg="${CFG_DIR}/${task}_${TGT_DOMAIN}.yaml"
    --models="${EXP_TRAIN_DIR}/${task}_${SRC_DOMAIN}/models"
    --log
  )
  SPARC train "${train_args[@]}"
  
  echo "=== Test: task=${task} Transfer Learning on domain=${TGT_DOMAIN} ==="
  test_args=(
    --input="${DATA_TEST_DIR}/${task_dir}_${TGT_DOMAIN}"
    --output="${EXP_TEST_DIR}/${task}_${TGT_DOMAIN}_transfer"
    --task="${task_cmd}"
    --cfg="${CFG_DIR}/${task}_${TGT_DOMAIN}.yaml"
    --models="${EXP_TRAIN_DIR}/${task}_${TGT_DOMAIN}_transfer/models"
    --log
    --save_qc
    --save_indiv
  )
  SPARC test "${test_args[@]}"
done
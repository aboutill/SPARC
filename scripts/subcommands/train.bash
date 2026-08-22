#!/usr/bin/env bash

# =============================================================================
# train.bash: part of sparc package.
#
# Train SPARC deep learning models via the sparc Docker image. Training is
# performed in n-fold cross-validation (CV) setup. Training can be performed
# across multiple-domain defined by separate input directories. Transfer
# learning can be activated by providing initialisation model weights. Training
# can be monitored via tensorboard server.
#
# Required arguments:
#  - Input directory/directories.
#  - Output directory.
#  - Deep learning task.
#  - Configuration file.
#
# Run with --help for usage.
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Default arguments
# =============================================================================
input=()
output=""
task=""
cfg=""
conf=false
folds=5
workers=8
models=false
verbose=false
log=true

# =============================================================================
# Do not edit
# =============================================================================
readonly DOCKER_IMG="aboutill/sparc:v1.0.0"
readonly DOCKER_INPUT_DIR="/mnt/input"
readonly DOCKER_OUTPUT_DIR="/mnt/output"
readonly DOCKER_CFG_DIR="/mnt/cfg"
readonly DOCKER_MODELS_DIR="/mnt/models"
readonly DOCKER_CONT="sparc_training"
readonly PORT="6006"
readonly SHM_SIZE="4960m"
readonly RELOAD_INT="30"

die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Command line helper
# =============================================================================
usage() {
  cat <<EOF
Usage: SPARC train [options]

Train deep learning models via the sparc Docker image. Training is
performed in n-fold cross-validation (CV) setup. Training can be performed
across multiple-domain defined by separate input directories. Transfer
learning can be activated by providing initialisation model weights. Training
can be monitored via tensorboard server.

Options:
  -i, --input <dir>     Input directory(ies)
  -o, --output <dir>    Output directory
  -t, --task <task>     Training task
  -c, --cfg <file>      Model configuration file
  -f, --conf <file>     Invocation configuration file
  -m, --models <dir>    Models directory
      --folds <int>     Number of CV folds (Default: 5)
      --workers <int>   Number of CPU workers/threads (Default: 8)
      --verbose         Enable verbose output
      --log             Enable logging
  -h, --help            Show this help message and exit

Examples:
  SPARC train --input /data/train/ --output /results/train/
EOF
}

# =============================================================================
# Command-line argument parsing
# =============================================================================
input_set_by_cli=false

require_value() {
  [[ -n "${2:-}" && "${2}" != -* ]] || die "missing or invalid value for $1"
}
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    -f|--conf)
      conf="${args[$((i+1))]:-}"
      [[ -n "${conf}" ]] || die "missing value for ${args[$i]}"
      ;;
    --conf=*)
      conf="${args[$i]#*=}"
      ;;
  esac
  i=$((i + 1))
done

if [[ "${conf}" != false ]]; then
  [[ -f "${conf}" ]] || die "conf file not found: ${conf}"
  source "${conf}"

  conf_dir=$(cd -- "$(dirname -- "$(readlink -f -- "${conf}")")/" &> /dev/null && pwd)
  resolve_path_conf() {
    local p="$1"
    if [[ "${p}" = /* ]]; then
      printf '%s\n' "${p}"
    else
      printf '%s\n' "${conf_dir}/${p}"
    fi
  }

  [[ ${#input[@]} -gt 0 ]] || die "missing value(s) for input"
  resolved_input=()
  for d in "${input[@]}"; do
    resolved_input+=("$(resolve_path_conf "${d}")")
  done
  input=("${resolved_input[@]}")

  [[ -n "${output}" ]] || die "missing value for output"
  output="$(resolve_path_conf "${output}")"

  [[ -n "${cfg}" ]] || die "missing value for cfg"
  cfg="$(resolve_path_conf "${cfg}")"

  if [[ "${models}" != false ]]; then
    [[ -n "${models}" ]] || die "missing value for models"
    models="$(resolve_path_conf "${models}")"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    -f|--conf)
      shift 2 ;;
    --conf=*)
      shift ;;
    -i|--input)
      require_value "$1" "${2:-}"
      if [[ "${input_set_by_cli}" == false ]]; then
        input=()
        input_set_by_cli=true
      fi
      input+=("$2")
      shift 2 ;;
    --input=*)
      require_value "$1" "${1#*=}"
      if [[ "${input_set_by_cli}" == false ]]; then
        input=()
        input_set_by_cli=true
      fi
      input+=("${1#*=}")
      shift ;;
    -o|--output)
      require_value "$1" "${2:-}"; output="$2"; shift 2 ;;
    --output=*)
      require_value "$1" "${1#*=}"; output="${1#*=}"; shift ;;
    -c|--cfg)
      require_value "$1" "${2:-}"; cfg="$2"; shift 2 ;;
    --cfg=*)
      require_value "$1" "${1#*=}"; cfg="${1#*=}"; shift ;;
    -t|--task)
      require_value "$1" "${2:-}"; task="$2"; shift 2 ;;
    --task=*)
      require_value "$1" "${1#*=}"; task="${1#*=}"; shift ;;
    -m|--models)
      require_value "$1" "${2:-}"; models="$2"; shift 2 ;;
    --models=*)
      require_value "$1" "${1#*=}"; models="${1#*=}"; shift ;;
    --folds)
      require_value "$1" "${2:-}"; folds="$2"; shift 2 ;;
    --folds=*)
      require_value "$1" "${1#*=}"; folds="${1#*=}"; shift ;;
    --workers)
      require_value "$1" "${2:-}"; workers="$2"; shift 2 ;;
    --workers=*)
      require_value "$1" "${1#*=}"; workers="${1#*=}"; shift ;;
    --verbose)
      verbose=true; shift ;;
    --log)
      log=true; shift ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

case "${task}" in
  segmentation|reorientation) ;;
  *) die "invalid task: ${mode} (expected segmentation|reorientation)" ;;
esac

# =============================================================================
# Validation
# =============================================================================
# Resolve a user-supplied path to an absolute path.
resolve_path() {
  local p="$1"
  if [[ "${p}" = /* ]]; then
    printf '%s\n' "${p}"
  else
    printf '%s\n' "$(cd -- "$(dirname -- "${p}")" &> /dev/null && pwd)/$(basename -- "${p}")"
  fi
}

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"

[[ ${#input[@]} -gt 0 ]] || die "missing value(s) for input"
[[ -n "${output}" ]] || die "missing value for output"
[[ -n "${cfg}" ]] || die "missing value for cfg"

host_output_dir="$(resolve_path "${output}")"

host_cfg_file="$(resolve_path "${cfg}")"
[[ -f "${host_cfg_file}" ]] || die "config file not found: ${host_cfg_file}"
host_cfg_dir="$(dirname -- "${host_cfg_file}")"
cfg_basename="$(basename "${host_cfg_file}")"

if [[ "${models}" != false ]]; then
  [[ -n "${models}" ]] || die "missing value for models"
  host_models_dir="$(resolve_path "${models}")"
  [[ -d "${host_models_dir}" ]] || die "models directory not found: ${host_models_dir}"
fi

mkdir -p "${host_output_dir}"

# =============================================================================
# Build pipeline command
# =============================================================================
tensorboard_cmd=(
  tensorboard
  --logdir "${DOCKER_OUTPUT_DIR}"
  --reload_multifile True
  --port="${PORT}"
  --reload_interval "${RELOAD_INT}"
  --bind_all
)

app_cmd=(
  sparc train
  --output "${DOCKER_OUTPUT_DIR}"
  --task "${task}"
  --cfg "${DOCKER_CFG_DIR}/${cfg_basename}"
  --folds "${folds}"
  --workers "${workers}"
)
[[ "${models}" != false ]] && app_cmd+=( --models "${DOCKER_MODELS_DIR}" )
[[ "${verbose}" == true ]] && app_cmd+=( --verbose )
[[ "${log}" == true ]] && app_cmd+=( --log )


docker_flags=(
  --user "$(id -u):$(id -g)"
  --volume "${host_cfg_dir}:${DOCKER_CFG_DIR}:ro"
  --volume "${host_output_dir}":"${DOCKER_OUTPUT_DIR}"
  -p "${PORT}:${PORT}"
  --shm-size="${SHM_SIZE}"
  --name "${DOCKER_CONT}"
  --gpus all
  --detach
)

[[ "${models}" != false ]] && docker_flags+=(
  --volume "${host_models_dir}:${DOCKER_MODELS_DIR}:ro"
)

for idx in "${!input[@]}"; do
  host_path="$(resolve_path "${input[$idx]}")"
  [[ -e "${host_path}" ]] || die "input not found: ${host_path}"
  docker_path="${DOCKER_INPUT_DIR}/${idx}"
  docker_flags+=( --volume "${host_path}:${docker_path}:ro" )
  app_cmd+=( --input "${docker_path}" )
done

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running sparc training via ${DOCKER_IMG} ..."

# Keep tensorboard listening
docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${tensorboard_cmd[@]}"

echo "Tensorboard server on host at http://localhost:${PORT}"

# Run python command
docker exec \
  "${DOCKER_CONT}" \
  "${app_cmd[@]}"

# Stop and remove container
docker stop "${DOCKER_CONT}"
docker rm "${DOCKER_CONT}"

echo "Done!"

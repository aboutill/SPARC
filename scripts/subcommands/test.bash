#!/usr/bin/env bash

# =============================================================================
# test.bash: part of sparc package.
#
# Test SPARC deep learning models via the sparc Docker image. Testing is
# performed in an ensemble setup.
#
# Required arguments:
#  - Input directory.
#  - Output directory.
#  - Deep leaning task.
#  - Models directory.
#  - Configuration file.
#
# Run with --help for usage.
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Default arguments
# =============================================================================
input=""
output=""
task=""
cfg=""
models=""
conf=false
workers=8
save_qc=false
save_indiv=false
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
readonly SHM_SIZE="4960m"

die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Command line helper
# =============================================================================
usage() {
  cat <<EOF
Usage: SPARC test [options]

Test deep learning models via the sparc Docker image. Testing is
performed in an ensemble setup.

Options:
  -i, --input <dir>     Input directory
  -o, --output <dir>    Output directory
  -t, --task <task>     Test task
  -c, --cfg <file>      Model configuration file
  -f, --conf <file>     Invocation configuration file
  -m, --models <dir>    Models directory
      --save_qc         Save quality control metrics
      --save_indiv      Save individual models outputs
      --workers <int>   Number of CPU workers/threads (Default: 8)
      --verbose         Enable verbose output
      --log             Enable logging
  -h, --help            Show this help message and exit

Examples:
  SPARC test --input /data/test/ --output /results/test/
EOF
}

# =============================================================================
# Command-line argument parsing
# =============================================================================
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

  [[ -n "${input}" ]] || die "missing value for input"
  input="$(resolve_path_conf "${input}")"
  
  [[ -n "${output}" ]] || die "missing value for output"
  output="$(resolve_path_conf "${output}")"
  
  [[ -n "${cfg}" ]] || die "missing value for cfg"
  cfg="$(resolve_path_conf "${cfg}")"
  
  [[ -n "${models}" ]] || die "missing value for models"
  models="$(resolve_path_conf "${models}")"
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
      require_value "$1" "${2:-}"; input="$2"; shift 2 ;;
    --input=*)
      require_value "$1" "${1#*=}"; input="${1#*=}"; shift ;;
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
    --workers)
      require_value "$1" "${2:-}"; workers="$2"; shift 2 ;;
    --workers=*)
      require_value "$1" "${1#*=}"; workers="${1#*=}"; shift ;;
    --save_qc) 
      save_qc=true; shift ;;
    --save_indiv) 
      save_indiv=true; shift ;;
    --verbose) 
      verbose=true; shift ;;
    --log)     
      log=true; shift ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

case "${task}" in
  segmentation|reorientation) ;;
  *) die "invalid task: ${task} (expected segmentation|reorientation)" ;;
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

[[ -n "${input}" ]] || die "missing value for input"
[[ -n "${output}" ]] || die "missing value for output"
[[ -n "${cfg}" ]] || die "missing value for cfg"
[[ -n "${models}" ]] || die "missing value for models"

host_input_dir="$(resolve_path "${input}")"
host_output_dir="$(resolve_path "${output}")"
[[ -d "${host_input_dir}" ]] || die "input not found: ${host_input_dir}"

host_cfg_file="$(resolve_path "${cfg}")"
[[ -f "${host_cfg_file}" ]] || die "config file not found: ${host_cfg_file}"
host_cfg_dir="$(dirname -- "${host_cfg_file}")"
cfg_basename="$(basename "${host_cfg_file}")"

host_models_dir="$(resolve_path "${models}")"
[[ -d "${host_models_dir}" ]] || die "models directory not found: ${host_models_dir}"

mkdir -p "${host_output_dir}"

# =============================================================================
# Build pipeline command
# =============================================================================
app_cmd=(
  sparc test
  --input "${DOCKER_INPUT_DIR}"
  --output "${DOCKER_OUTPUT_DIR}"
  --task "${task}"
  --cfg "${DOCKER_CFG_DIR}/${cfg_basename}"
  --models "${DOCKER_MODELS_DIR}"
  --workers "${workers}"
)
[[ "${save_qc}" == true ]] && app_cmd+=( --save_qc )
[[ "${save_indiv}" == true ]] && app_cmd+=( --save_indiv )
[[ "${verbose}" == true ]] && app_cmd+=( --verbose )
[[ "${log}" == true ]] && app_cmd+=( --log )


docker_flags=( 
  --user "$(id -u):$(id -g)"
  --volume "${host_input_dir}:${DOCKER_INPUT_DIR}:ro"
  --volume "${host_cfg_dir}:${DOCKER_CFG_DIR}:ro"
  --volume "${host_output_dir}":"${DOCKER_OUTPUT_DIR}"
  --volume "${host_models_dir}":"${DOCKER_MODELS_DIR}"
  --shm-size="${SHM_SIZE}"
  --gpus all
)

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running sparc test via ${DOCKER_IMG} ..."

# Run python command
docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${app_cmd[@]}"
  
echo "Done!"
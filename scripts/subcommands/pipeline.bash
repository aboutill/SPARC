#!/usr/bin/env bash

# =============================================================================
# pipeline.bash: part of sparc package.
#
# Run the Slice-to-volume Pipeline for Automated Reconstruction of
# gated Cardiac cine (SPARC) via the sparc Docker image.
#
# Required arguments:
#  - Input DICOM directory.
#  - Output directory.
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
batch=false
cfg=false
conf=false
models_chest_seg=false
models_chest_seg_cfg=false
models_heart_seg=false
models_heart_seg_cfg=false
models_reo=false
models_reo_cfg=false
mode="semi_auto"
gui_mode="docker"
manual_stack_review=false
file_prefix=false
verbose=false
profile=false
log=false
debug=false

# =============================================================================
# Do not edit
# =============================================================================
readonly DOCKER_IMG="aboutill/sparc:v1.0.0"
readonly DOCKER_INPUT_DIR="/mnt/input"
readonly DOCKER_OUTPUT_DIR="/mnt/output"
readonly DOCKER_CFG_DIR="/mnt/cfg"
readonly DOCKER_MODELS_DIR="/mnt/models"

die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Command line helper
# =============================================================================
usage() {
  cat <<EOF
Usage: SPARC pipeline [options]

Run the SPARC pipeline in a Docker container.

Options:
  -i, --input <dir>                   Input DICOM directory
  -o, --output <dir>                  Output directory
  -b, --batch                         Enable batch processing; disables subject processing
  -m, --mode <mode>                   Pipeline mode: manual | semi_auto | monitored_auto | full_auto (default: ${mode})
  -c, --cfg <file>                    Pipeline configuration file
  -f, --conf <file>                   Invocation configuration file
      --models_chest_seg <dir>        Chest segmentation models directory
      --models_chest_seg_cfg <file>   Chest segmentation models configuration file
      --models_heart_seg <dir>        Heart segmentation models directory
      --models_heart_seg_cfg <file>   Heart segmentation models configuration file
      --models_reo <dir>              Reorientation models directory
      --models_reo_cfg <file>         Reorientation models configuration file
      --gui_mode <gui_mode>           Pipeline GUI mode: docker | native (default: ${gui_mode})
      --manual_stack_review           Enable manual stack review through GUI, disabled in full_auto mode
      --file_prefix                   Output files prefix; disabled in batch mode
      --verbose                       Enable verbose output
      --profile                       Enable profiling
      --log                           Enable logging
      --debug                         Enable debug mode
  -h, --help                          Show this help message and exit

Examples:
  SPARC pipeline --input /data/fetal/raw --output /data/fetal/out
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
  [[ -n "${output}" ]] || die "missing value for output"

  input="$(resolve_path_conf "${input}")"
  output="$(resolve_path_conf "${output}")"
  if [[ "${cfg}" != false ]]; then
    [[ -n "${cfg}" ]] || die "missing value for cfg"
    cfg="$(resolve_path_conf "${cfg}")"
  fi
  
  if [[ "${models_chest_seg}" != false ]]; then
    [[ -n "${models_chest_seg}" ]] || die "missing value for models_chest_seg"
    models_chest_seg="$(resolve_path_conf "${models_chest_seg}")"
  fi
  if [[ "${models_heart_seg}" != false ]]; then
    [[ -n "${models_heart_seg}" ]] || die "missing value for models_heart_seg"
    models_heart_seg="$(resolve_path_conf "${models_heart_seg}")"
  fi
  if [[ "${models_reo}" != false ]]; then
    [[ -n "${models_reo}" ]] || die "missing value for models_reo"
    models_reo="$(resolve_path_conf "${models_reo}")"
  fi
  
  if [[ "${models_chest_seg_cfg}" != false ]]; then
    [[ -n "${models_chest_seg_cfg}" ]] || die "missing value for models_chest_seg_cfg"
    models_chest_seg_cfg="$(resolve_path_conf "${models_chest_seg_cfg}")"
  fi
  if [[ "${models_heart_seg_cfg}" != false ]]; then
    [[ -n "${models_heart_seg_cfg}" ]] || die "missing value for models_heart_seg_cfg"
    models_heart_seg_cfg="$(resolve_path_conf "${models_heart_seg_cfg}")"
  fi
  if [[ "${models_reo_cfg}" != false ]]; then
    [[ -n "${models_reo_cfg}" ]] || die "missing value for models_reo_cfg"
    models_reo_cfg="$(resolve_path_conf "${models_reo_cfg}")"
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
      require_value "$1" "${2:-}"; input="$2"; shift 2 ;;
    --input=*)
      require_value "$1" "${1#*=}"; input="${1#*=}"; shift ;;
    -o|--output)
      require_value "$1" "${2:-}"; output="$2"; shift 2 ;;
    --output=*)
      require_value "$1" "${1#*=}"; output="${1#*=}"; shift ;;
    -b|--batch)
      batch=true; shift ;;
    -c|--cfg)
      require_value "$1" "${2:-}"; cfg="$2"; shift 2 ;;
    --cfg=*)
      require_value "$1" "${1#*=}"; cfg="${1#*=}"; shift ;;
    -m|--mode)
      require_value "$1" "${2:-}"; mode="$2"; shift 2 ;;
    --mode=*)
      require_value "$1" "${1#*=}"; mode="${1#*=}"; shift ;;
    --models_chest_seg)
      require_value "$1" "${2:-}"; models_chest_seg="$2"; shift 2 ;;
    --models_chest_seg=*)
      require_value "$1" "${1#*=}"; models_chest_seg="${1#*=}"; shift ;;
    --models_chest_seg_cfg)
      require_value "$1" "${2:-}"; models_chest_seg_cfg="$2"; shift 2 ;;
    --models_chest_seg_cfg=*)
      require_value "$1" "${1#*=}"; models_chest_seg_cfg="${1#*=}"; shift ;;
    --models_heart_seg)
      require_value "$1" "${2:-}"; models_heart_seg="$2"; shift 2 ;;
    --models_heart_seg=*)
      require_value "$1" "${1#*=}"; models_heart_seg="${1#*=}"; shift ;;
    --models_heart_seg_cfg)
      require_value "$1" "${2:-}"; models_heart_seg_cfg="$2"; shift 2 ;;
    --models_heart_seg_cfg=*)
      require_value "$1" "${1#*=}"; models_heart_seg_cfg="${1#*=}"; shift ;;
    --models_reo)
      require_value "$1" "${2:-}"; models_reo="$2"; shift 2 ;;
    --models_reo=*)
      require_value "$1" "${1#*=}"; models_reo="${1#*=}"; shift ;;
    --models_reo_cfg)
      require_value "$1" "${2:-}"; models_reo_cfg="$2"; shift 2 ;;
    --models_reo_cfg=*)
      require_value "$1" "${1#*=}"; models_reo_cfg="${1#*=}"; shift ;;
    --gui_mode)
      require_value "$1" "${2:-}"; gui_mode="$2"; shift 2 ;;
    --gui_mode=*)
      require_value "$1" "${1#*=}"; gui_mode="${1#*=}"; shift ;;
    --file_prefix)
      require_value "$1" "${2:-}"; file_prefix="$2"; shift 2 ;;
    --file_prefix=*)
      require_value "$1" "${1#*=}"; file_prefix="${1#*=}"; shift ;;
    --manual_stack_review) 
      manual_stack_review=true; shift ;;
    --verbose) 
      verbose=true; shift ;;
    --profile) 
      profile=true; shift ;;
    --log)     
      log=true; shift ;;
    --debug)   
      debug=true; shift ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

case "${mode}" in
  manual|semi_auto|monitored_auto|full_auto) ;;
  *) die "invalid mode: ${mode} (expected manual|semi_auto|monitored_auto|full_auto)" ;;
esac

case "${gui_mode}" in
  docker|native) ;;
  *) die "invalid gui_mode: gui_modemode} (expected docker|native)" ;;
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

host_input_dir="$(resolve_path "${input}")"
host_output_dir="$(resolve_path "${output}")"
[[ -d "${host_input_dir}" ]] || die "input not found: ${host_input_dir}"

if [[ "${cfg}" != false ]]; then
  [[ -n "${cfg}" ]] || die "missing value for cfg"
  host_cfg_file="$(resolve_path "${cfg}")"
  [[ -f "${host_cfg_file}" ]] || die "config file not found: ${host_cfg_file}"
  host_cfg_dir="$(dirname -- "${host_cfg_file}")"
  cfg_basename="$(basename "${host_cfg_file}")"
fi

mkdir -p "${host_output_dir}"

# =============================================================================
# Build pipeline command
# =============================================================================
app_cmd=(
  sparc pipeline
  --input "${DOCKER_INPUT_DIR}"
  --output "${DOCKER_OUTPUT_DIR}"
  --mode "${mode}"
  --gui_mode "${gui_mode}"
)
[[ "${file_prefix}" != false ]] && app_cmd+=( --file_prefix "${file_prefix}" )
[[ "${cfg}" != false ]] && app_cmd+=( --cfg "${DOCKER_CFG_DIR}/${cfg_basename}" )
[[ "${batch}" == true ]] && app_cmd+=( --batch )
[[ "${manual_stack_review}" == true ]] && app_cmd+=( --manual_stack_review )
[[ "${verbose}" == true ]] && app_cmd+=( --verbose )
[[ "${profile}" == true ]] && app_cmd+=( --profile )
[[ "${debug}" == true ]] && app_cmd+=( --debug )
[[ "${log}" == true ]] && app_cmd+=( --log )

docker_flags=(
  --rm
  --user "$(id -u):$(id -g)"
  --volume "${host_input_dir}:${DOCKER_INPUT_DIR}:ro"
  --volume "${host_output_dir}:${DOCKER_OUTPUT_DIR}"
  --gpus all
)
if [[ "${mode}" != "full_auto" && "${gui_mode}" == "docker" ]]; then
  docker_flags+=(
  --volume /tmp/.X11-unix:/tmp/.X11-unix
  --security-opt apparmor=unconfined
  --ipc=host
  --env DISPLAY="${DISPLAY}"
  --env QT_X11_NO_MITSHM=1
  --env LIBGL_ALWAYS_SOFTWARE=1
  --env QT_QPA_PLATFORM=xcb
)
fi
[[ "${cfg}" != false ]] && docker_flags+=(
  --volume "${host_cfg_dir}:${DOCKER_CFG_DIR}:ro"
)
[[ -t 0 && -t 1 ]] && docker_flags+=( -it )

# =============================================================================
# Model arguments
# =============================================================================
mount_component() {
  local dir_var="$1" cfg_var="$2" component_name="$3"
  local host_dir="${!dir_var}" host_cfg="${!cfg_var}"

  if [[ "${host_dir}" != false ]]; then
    host_dir="$(resolve_path "${host_dir}")"
    [[ -d "${host_dir}" ]] || die "${component_name} models directory not found: ${host_dir}"
    local docker_dir="${DOCKER_MODELS_DIR}/${component_name}"
    docker_flags+=( --volume "${host_dir}:${docker_dir}:ro" )
    app_cmd+=( --models_${component_name} "${docker_dir}" )
  fi

  if [[ "${host_cfg}" != false ]]; then
    host_cfg="$(resolve_path "${host_cfg}")"
    [[ -f "${host_cfg}" ]] || die "${component_name} model config not found: ${host_cfg}"
    local docker_cfg_dir="${DOCKER_MODELS_DIR}/${component_name}_cfg"
    docker_flags+=( --volume "$(dirname "${host_cfg}"):${docker_cfg_dir}:ro" )
    app_cmd+=( --models_${component_name}_cfg "${docker_cfg_dir}/$(basename "${host_cfg}")" )
  fi
}

if [[ "${models_chest_seg}" != false && "${models_chest_seg_cfg}" != false ]]; then
  mount_component models_chest_seg models_chest_seg_cfg chest_seg
fi
if [[ "${models_heart_seg}" != false && "${models_heart_seg_cfg}" != false ]]; then
  mount_component models_heart_seg models_heart_seg_cfg heart_seg
fi
if [[ "${models_reo}" != false && "${models_reo_cfg}" != false ]]; then
  mount_component models_reo models_reo_cfg reo
fi

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running SPARC pipeline via ${DOCKER_IMG} ..."

if [[ "${mode}" != "full_auto" ]]; then
  trap 'xhost -local:docker' EXIT
  xhost +local:docker
fi

docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${app_cmd[@]}"

echo "Done!"
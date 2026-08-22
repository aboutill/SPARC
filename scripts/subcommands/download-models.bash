#!/usr/bin/env bash
# =============================================================================
# download-models.bash: part of sparc package.
#
# Download pretrained model weights from Hugging Face for the chest
# segmentation, heart segmentation, and reorientation components.
# For use as a transfer-learning starting point with
# `train.bash --models <path>`.
#
# =============================================================================

set -euo pipefail

readonly HF_ORG="aboutill"
readonly COMPONENTS=("chest-segmentation" "heart-segmentation" "reorientation")
readonly VARIANTS=("philips" "siemens" "siemens_transfer" "joint")
readonly DOCKER_IMG="aboutill/sparc:v1.0.0"
readonly DOCKER_MODELS_DIR="/mnt/models"

output=""
component="all"
variant="siemens_transfer"
force=false

declare -A HF_REPO_NAMES=(
  ["chest-segmentation"]="sparc-chest-segmentation"
  ["heart-segmentation"]="sparc-heart-segmentation"
  ["reorientation"]="sparc-reorientation"
)


die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Command line helper
# =============================================================================
usage() {
  cat <<EOF
Usage: SPARC download-models [options]

Download pretrained model weights and configs.

Options:
  -o, --output              Location of downloaded models
  -c, --component <name>    One of: ${COMPONENTS[*]}, or "all" (default: all)
  -v, --variant <name>      One of: ${VARIANTS[*]}, or "all" (default: siemens_transfer)
  -f, --force               Re-download even if already present locally
  -h, --help                Show this help message and exit

Examples:
  SPARC download-models --output /models
      Download the default (siemens_transfer) variant of every component.

  SPARC download-models --output /models --component reorientation --variant all
      Download every reorientation model variant.
EOF
}

require_value() {
  [[ -n "${2:-}" && "${2}" != -* ]] || die "missing or invalid value for $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    -o|--output)
      require_value "$1" "${2:-}"; output="$2"; shift 2 ;;
    --output=*)
      require_value "$1" "${1#*=}"; output="${1#*=}"; shift ;;
    -c|--component)
      require_value "$1" "${2:-}"; component="$2"; shift 2 ;;
    --component=*)
      require_value "$1" "${1#*=}"; component="${1#*=}"; shift ;;
    -v|--variant)
      require_value "$1" "${2:-}"; variant="$2"; shift 2 ;;
    --variant=*)
      require_value "$1" "${1#*=}"; variant="${1#*=}"; shift ;;
    -f|--force)
      force=true; shift ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

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
[[ -n "${output}" ]] || die "missing value for output"
host_output="$(resolve_path "${output}")"

if [[ "${component}" != "all" ]]; then
  match=false
  for c in "${COMPONENTS[@]}"; do [[ "${c}" == "${component}" ]] && match=true; done
  [[ "${match}" == true ]] || die "invalid --component '${component}' (expected: ${COMPONENTS[*]}, or all)"
fi
if [[ "${variant}" != "all" ]]; then
  match=false
  for v in "${VARIANTS[@]}"; do [[ "${v}" == "${variant}" ]] && match=true; done
  [[ "${match}" == true ]] || die "invalid --variant '${variant}' (expected: ${VARIANTS[*]}, or all)"
fi

# =============================================================================
# Download one component/variant from hugging face.
# =============================================================================
download_one() {
  local comp="$1" var="$2"
  local repo_name="${HF_REPO_NAMES[${comp}]}"
  local repo_id="${HF_ORG}/${repo_name}"
  local dest_dir="${host_output}/${comp}/${var}"

  if [[ -d "${dest_dir}" && "${force}" != true ]]; then
    echo "  [skip] ${comp}/${var} already present (use --force to re-download)"
    return
  fi

  echo "  Downloading ${repo_id} @ ${var} -> ${dest_dir}"
  mkdir -p "${dest_dir}"

  app_cmd=(
    hf download
    "${repo_id}"
    --repo-type model
    --revision "${var}"
    --local-dir "${DOCKER_MODELS_DIR}/${comp}/${var}"
    --include '*.pth'
    --include '*.yaml'
  )
  docker_flags=(
    --rm
    --user "$(id -u):$(id -g)"
    --volume "${host_output}:${DOCKER_MODELS_DIR}"
  )
  docker run \
    "${docker_flags[@]}" \
    "${DOCKER_IMG}" \
    "${app_cmd[@]}" \
  || die "failed to download ${repo_id} @ ${var} does this repo/revision exist?"

  echo "  Done: ${comp}/${var}"
}


# =============================================================================
# Resolve which (component, variant) pairs to fetch, and run.
# =============================================================================
components_to_fetch=()
if [[ "${component}" == "all" ]]; then
  components_to_fetch=("${COMPONENTS[@]}")
else
  components_to_fetch=("${component}")
fi

variants_to_fetch=()
if [[ "${variant}" == "all" ]]; then
  variants_to_fetch=("${VARIANTS[@]}")
else
  variants_to_fetch=("${variant}")
fi

echo "Downloading models to ${host_output}"
echo "Components: ${components_to_fetch[*]}"
echo "Variants: ${variants_to_fetch[*]}"
echo

for comp in "${components_to_fetch[@]}"; do
  for var in "${variants_to_fetch[@]}"; do
    download_one "${comp}" "${var}"
  done
done

echo
echo "Done."

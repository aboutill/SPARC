#!/usr/bin/env bash
# =============================================================================
# notebooks.bash: part of sparc package.
#
# Run the Jupyter notebooks interactively via the sparc Docker image.
#
# Run with --help for usage.
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Default arguments
# =============================================================================
workdir=""

# =============================================================================
# Do not edit
# =============================================================================
readonly DOCKER_IMG="aboutill/sparc:v1.0.0"
readonly DOCKER_WORKDIR="/mnt/workdir"
readonly PORT=8888

die() { echo "Error: $*" >&2; exit 1; }

# =============================================================================
# Command line helper
# =============================================================================
usage() {
  cat <<EOF
Usage: SPARC notebook [options]

Run the SPARC notebooks in a Docker container.

Options:
  -w, --workdir <dir>  Input notebook directory
  -h, --help           Show this help message and exit

Examples:
  SPARC notebook --workdir /workdir
EOF
}

# =============================================================================
# Command-line argument parsing
# =============================================================================
require_value() {
  [[ -n "${2:-}" && "${2}" != -* ]] || die "missing or invalid value for $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    -w|--workdir)
      require_value "$1" "${2:-}"; workdir="$2"; shift 2 ;;
    --workdir=*)
      require_value "$1" "${1#*=}"; workdir="${1#*=}"; shift ;;
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
[[ -n "${workdir}" ]] || die "missing value for workdir"
host_workdir="$(resolve_path "${workdir}")"
[[ -d "${host_workdir}" ]] || die "workdir not found: ${host_workdir}"

# =============================================================================
# Build command
# =============================================================================
app_cmd=(
  jupyter notebook 
  --ip=0.0.0.0
  --port="${PORT}"
  --no-browser
)

docker_flags=( 
  --rm 
  --user "$(id -u):$(id -g)" 
  -p "${PORT}":"${PORT}" 
  --volume "${host_workdir}:${DOCKER_WORKDIR}"
)
[[ -t 0 && -t 1 ]] && docker_flags+=( -it )

# =============================================================================
# Run Docker application
# =============================================================================
echo "Running notebooks via ${DOCKER_IMG} ..."

docker run \
  "${docker_flags[@]}" \
  "${DOCKER_IMG}" \
  "${app_cmd[@]}"
  
echo "Done!"

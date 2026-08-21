#!/usr/bin/env bash

# =============================================================================
# runSPARC.bash: part of sparc package.
#
# Run the Slice-to-volume Pipeline for Automated Reconstruction of
# gated Cardiac cine (SPARC) via the sparc Docker image.
# KCL internal wrapper.
#
# Required arguments:
#  - Input DICOM directory.
#  - Output directory.
#
# =============================================================================

set -euo pipefail

# =============================================================================
# USER INPUTS
# =============================================================================
# Required arguments, relative to app root
input="data"
output="recon"

# Select either one of the two options
subID=false # Set subject ID or false
batch=true # Batch processing

# Optional arguments
mode="semi_auto" # Options: "manual", "semi_auto", "monitored_auto", "full_auto"
# Activate manual stack review through GUI (stack exclusion and cardiac synchronisation)
manual_stack_review=false # [Incompatible with mode="full_auto"]
cfg="cfg/pipeline.yaml" # relative to app root,
                        # false to use default pipeline configuration

# Behavious
verbose=false
profile=false
log=true
debug=false

# =============================================================================
# Do not edit
# =============================================================================
readonly HOST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)

command -v SPARC >/dev/null 2>&1 || {
  echo "Error: 'SPARC' not found on PATH. Run 'make install' from the" >&2
  echo "SPARC repo once per machine (see the repo's Makefile)." >&2
  exit 1
}

if [[ "${subID}" != false && "${batch}" == true ]]; then
  echo "Error: set exactly one of subID or batch=true, not both." >&2
  exit 1
fi

if [ "${subID}" != false ] ; then
    input="${input}/${subID}"
    output="${output}/${subID}"
fi
sparc_args=(
  --input="${HOST_DIR}/${input}"
  --output="${HOST_DIR}/${output}"
  --mode="${mode}"
)
[[ "${subID}" != false ]] && sparc_args+=( --file_prefix="${subID}" )
[[ "${cfg}" != false ]] && sparc_args+=( --cfg="${HOST_DIR}/${cfg}" )
[[ "${batch}" == true ]] && sparc_args+=( --batch )
[[ "${verbose}" == true ]] && sparc_args+=( --verbose )
[[ "${profile}" == true ]] && sparc_args+=( --profile )
[[ "${debug}" == true ]] && sparc_args+=( --debug )
[[ "${log}" == true ]] && sparc_args+=( --log )

SPARC pipeline "${sparc_args[@]}"
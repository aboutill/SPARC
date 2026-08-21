#!/usr/bin/env bash

# =============================================================================
# SPARC.bash: part of sparc package.
#
# Bash scripts dispatcher.
#
# =============================================================================
set -euo pipefail

SUBCOMMAND_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/subcommands" &> /dev/null && pwd)"

usage() {
  cat <<EOF
Usage: SPARC <subcommand> [options]

Subcommands:
  pipeline                 Run the end-to-end SPARC pipeline
  train                    Train a deep learning component
  test                     Evaluate a trained component on new data
  download-models          Download pretrained models (weights and cfg files)
  notebooks                Run notebooks environment
  help                     Show this message

Run 'SPARC <subcommand> --help' for subcommand-specific options.
EOF
}

[[ $# -gt 0 ]] || { usage; exit 1; }

subcommand="$1"; shift

case "${subcommand}" in
  -h|--help|help)
    usage; exit 0 ;;
  pipeline)
    exec "${SUBCOMMAND_DIR}/pipeline.bash" "$@" ;;
  train)
    exec "${SUBCOMMAND_DIR}/train.bash" "$@" ;;
  test)
    exec "${SUBCOMMAND_DIR}/test.bash" "$@" ;;
  download-models)
    exec "${SUBCOMMAND_DIR}/download-models.bash" "$@" ;;
  notebooks)
    exec "${SUBCOMMAND_DIR}/notebooks.bash" "$@" ;;
  *)
    echo "Error: unknown subcommand '${subcommand}' (see 'SPARC help')" >&2
    exit 1 ;;
esac

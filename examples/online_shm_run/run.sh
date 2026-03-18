#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PyCWB online search — shared-memory (llhoft) run script
#
# Prerequisites on the server:
#   1. PyCWB installed and on PATH  (pycwb/pycwb_search/pycwb_show)
#   2. /dev/shm/kafka/{H1,L1}/ directories populated by the low-latency
#      data writer (e.g. GWDataFind / LDAStools writer)
#   3. GWF files following the naming convention:
#        H-H1_llhoft-<GPS>-1.gwf   →  /dev/shm/kafka/H1/
#        L-L1_llhoft-<GPS>-1.gwf   →  /dev/shm/kafka/L1/
#
# Usage:
#   bash run.sh [--workers N] [--log-level LEVEL]
#
# Options:
#   --workers N       Number of parallel analysis workers (default: 4)
#   --log-level LEVEL DEBUG|INFO|WARNING|ERROR          (default: INFO)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/user_parameters.yaml"
WORK_DIR="${SCRIPT_DIR}/output"
WORKERS=4
LOG_LEVEL="INFO"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)  WORKERS="$2";   shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "${WORK_DIR}"

echo "──────────────────────────────────────────────"
echo " PyCWB online search (shm)"
echo " Config   : ${CONFIG}"
echo " Work dir : ${WORK_DIR}"
echo " Workers  : ${WORKERS}"
echo " Log level: ${LOG_LEVEL}"
echo " Data src : /dev/shm/kafka/{H1,L1}/"
echo "──────────────────────────────────────────────"

# Verify data directories exist before starting
for IFO in H1 L1; do
  DIR="/dev/shm/kafka/${IFO}"
  if [[ ! -d "${DIR}" ]]; then
    echo "WARNING: ${DIR} does not exist yet — will wait for files at runtime"
  else
    COUNT=$(find "${DIR}" -name '*.gwf' | wc -l | tr -d ' ')
    echo "  ${DIR}: ${COUNT} GWF file(s) present"
  fi
done

echo ""
echo "Starting online search... (Ctrl-C to stop)"
exec pycwb online "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --n-workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"

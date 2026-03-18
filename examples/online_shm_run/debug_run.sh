#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# debug_run.sh — start fake data generator + online pipeline for local testing
#
# Steps:
#   1. Pre-populates /dev/shm/kafka/{H1,L1}/ with the first GPS_PREFILL
#      seconds of frames as fast as possible (--no-realtime).
#   2. Launches the fake data generator in real-time mode in the background
#      so it continues writing new 1-second frames as the pipeline consumes them.
#   3. Starts the pycwb online pipeline against the shm data source.
#
# Usage:
#   bash debug_run.sh [--gps-start GPS] [--workers N] [--log-level LEVEL]
#                     [--no-inject] [--prefill N]
#
# Options:
#   --gps-start GPS   GPS integer start time (default: current GPS)
#   --workers N       Online pipeline worker count (default: 4)
#   --log-level LEVEL Pipeline log level: DEBUG|INFO|WARNING (default: INFO)
#   --no-inject       Skip the CBC injection
#   --prefill N       Pre-fill N seconds before going real-time (default: 120)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/user_parameters.yaml"
WORK_DIR="${SCRIPT_DIR}/output"
GENERATOR="${SCRIPT_DIR}/fake_data_generator.py"

# Python interpreter — prefer pycwb-dev-py13 conda env, fall back to system python3
PYTHON="${PYCWB_PYTHON:-/Users/yumengxu/miniforge3/envs/pycwb-dev-py13/bin/python3}"

# Defaults
GPS_START=""
WORKERS=4
LOG_LEVEL="INFO"
INJECT_FLAG=""
PREFILL=120
DURATION=3600
SHM_BASE="/dev/shm/kafka"
CHANNEL="GDS-CALIB_STRAIN_CLEAN_C00"
IFOS="H1 L1"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gps-start)  GPS_START="$2"; shift 2 ;;
    --workers)    WORKERS="$2";   shift 2 ;;
    --log-level)  LOG_LEVEL="$2"; shift 2 ;;
    --no-inject)  INJECT_FLAG="--no-injection"; shift ;;
    --prefill)    PREFILL="$2";   shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Resolve GPS start
if [[ -z "${GPS_START}" ]]; then
  # try astropy, fall back to rough calculation
  GPS_START=$("${PYTHON}" -c "
try:
    from astropy.time import Time
    print(int(Time.now().gps))
except ImportError:
    import time, calendar
    GPS_EPOCH = calendar.timegm((1980,1,6,0,0,0,0,0,0))
    print(int(time.time() - GPS_EPOCH + 18))
" 2>/dev/null) || GPS_START=1257894000
fi

mkdir -p "${WORK_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo " PyCWB online debug run"
echo " GPS start : ${GPS_START}"
echo " Pre-fill  : ${PREFILL} s"
echo " Duration  : ${DURATION} s"
echo " Workers   : ${WORKERS}"
echo " Log level : ${LOG_LEVEL}"
echo " SHM base  : ${SHM_BASE}"
echo " Work dir  : ${WORK_DIR}"
echo "═══════════════════════════════════════════════════════════"

# ── Step 1: Pre-populate SHM as fast as possible ─────────────────────────────
echo ""
echo "▶ Pre-filling ${PREFILL} s of SHM frames (fast)…"
"${PYTHON}" "${GENERATOR}" \
    --gps-start "${GPS_START}" \
    --duration  "${DURATION}" \
    --sample-rate 16384 \
    --ifos ${IFOS} \
    --channel "${CHANNEL}" \
    --shm-base "${SHM_BASE}" \
    --no-realtime \
    --max-frames "${PREFILL}" \
    --seed 42 \
    ${INJECT_FLAG}

echo "  Pre-fill complete."

# ── Step 2: Continue writing in real-time in the background ──────────────────
echo ""
echo "▶ Starting real-time fake data generator in background…"
REMAINING=$((DURATION - PREFILL))
REALTIME_GPS_START=$((GPS_START + PREFILL))

"${PYTHON}" "${GENERATOR}" \
    --gps-start "${REALTIME_GPS_START}" \
    --duration  "${REMAINING}" \
    --sample-rate 16384 \
    --ifos ${IFOS} \
    --channel "${CHANNEL}" \
    --shm-base "${SHM_BASE}" \
    --realtime \
    --max-frames "${REMAINING}" \
    --seed 42 \
    ${INJECT_FLAG} \
    > "${WORK_DIR}/fake_data_generator.log" 2>&1 &

GENERATOR_PID=$!
echo "  Generator PID: ${GENERATOR_PID}  (log: ${WORK_DIR}/fake_data_generator.log)"

# ── Step 3: Launch online pipeline ───────────────────────────────────────────
echo ""
echo "▶ Starting pycwb online pipeline…  (Ctrl-C to stop both)"

# Trap to clean up generator on exit
cleanup() {
  echo ""
  echo "Stopping fake data generator (PID ${GENERATOR_PID})…"
  kill "${GENERATOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

pycwb online "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --n-workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"

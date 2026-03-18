#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# debug_run.sh — real-time fake data generator + online pipeline for local testing
#
# The fake data generator runs in the background at real-time speed (1 frame/s),
# writing to /tmp/fake_stream.  The pipeline starts immediately alongside it.
# Generator GPS start = current_gps - segment_duration so the first analysis
# window is available with minimal latency.
#
# Usage:
#   bash debug_run.sh [--workers N] [--log-level LEVEL] [--no-inject]
#                     [--duration N]
#
# Options:
#   --workers N       Online pipeline worker count (default: 4)
#   --log-level LEVEL Pipeline log level: DEBUG|INFO|WARNING (default: INFO)
#   --no-inject       Skip the CBC injection
#   --duration N      Seconds of fake data for the generator to produce (default: 3600)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/user_parameters_debug.yaml"
WORK_DIR="${SCRIPT_DIR}/output"
GENERATOR="${SCRIPT_DIR}/fake_data_generator.py"

# Python interpreter — prefer pycwb-dev-py13 conda env, fall back to system python3
PYTHON="${PYCWB_PYTHON:-/Users/yumengxu/miniforge3/envs/pycwb-dev-py13/bin/python3}"
# pycwb CLI — derived from the same conda env directory as PYTHON
PYCWB_BIN="${PYCWB_BIN:-$(dirname "${PYTHON}")/pycwb}"

# Defaults
WORKERS=4
LOG_LEVEL="INFO"
INJECT_FLAG=""
DURATION=3600         # seconds of fake data to generate
SEGMENT_DURATION=60   # must match online_segment_duration in config
SHM_BASE="/tmp/fake_stream"
CHANNEL="GDS-CALIB_STRAIN_CLEAN_C00"
DQ_CHANNEL="DMT-DQ_VECTOR"  # written alongside strain; use '' to disable
IFOS="H1 L1"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)    WORKERS="$2";   shift 2 ;;
    --log-level)  LOG_LEVEL="$2"; shift 2 ;;
    --no-inject)  INJECT_FLAG="--no-injection"; shift ;;
    --duration)   DURATION="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Get current GPS time — prefer lal, fall back to astropy, then manual calc
GPS_NOW=$("${PYTHON}" -c "
try:
    from lal import gpstime
    print(int(gpstime.gps_time_now()))
except ImportError:
    try:
        from astropy.time import Time
        print(int(Time.now().gps))
    except ImportError:
        import time, calendar
        GPS_EPOCH = calendar.timegm((1980,1,6,0,0,0,0,0,0))
        print(int(time.time() - GPS_EPOCH + 18))
" 2>/dev/null) || GPS_NOW=1257894000

# Generator starts from now; pipeline also starts from now.
# The first segment is emitted after SEGMENT_DURATION seconds of data accumulate.
GENERATOR_GPS_START=${GPS_NOW}

mkdir -p "${WORK_DIR}"

# Clear stale data from any previous run
echo "Clearing stale data from ${SHM_BASE}…"
rm -rf "${SHM_BASE}"
mkdir -p "${SHM_BASE}"

echo "═══════════════════════════════════════════════════════════"
echo " PyCWB online debug run (real-time)"
echo " GPS now / start     : ${GPS_NOW}"
echo " Generator GPS start : ${GENERATOR_GPS_START} (= GPS now)"
echo " Generator duration  : ${DURATION} s"
echo " Workers             : ${WORKERS}"
echo " Log level           : ${LOG_LEVEL}"
echo " SHM base            : ${SHM_BASE}"
echo " Work dir            : ${WORK_DIR}"
echo "═══════════════════════════════════════════════════════════"

# ── Launch real-time fake data generator in the background ───────────────────
echo ""
echo "▶ Starting real-time fake data generator in background…"
echo "  (GPS ${GENERATOR_GPS_START}, 1 frame/s, ${DURATION} s total)"

"${PYTHON}" "${GENERATOR}" \
    --gps-start "${GENERATOR_GPS_START}" \
    --duration  "${DURATION}" \
    --sample-rate 16384 \
    --ifos ${IFOS} \
    --channel "${CHANNEL}" \
    --dq-channel "${DQ_CHANNEL}" \
    --shm-base "${SHM_BASE}" \
    --realtime \
    --seed 42 \
    ${INJECT_FLAG} \
    > "${WORK_DIR}/fake_data_generator.log" 2>&1 &

GENERATOR_PID=$!
echo "  Generator PID: ${GENERATOR_PID}  (log: ${WORK_DIR}/fake_data_generator.log)"

# Trap to stop generator on exit
cleanup() {
  echo ""
  echo "Stopping fake data generator (PID ${GENERATOR_PID})…"
  kill "${GENERATOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Launch online pipeline ────────────────────────────────────────────────────
echo ""
echo "▶ Starting pycwb online pipeline…  (Ctrl-C to stop both)"

"${PYCWB_BIN}" online "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --n-workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"

"""
Integration test for the online SHM pipeline.
Run from the online_shm_run directory:
    conda run -n pycwb-dev-py13 python _test_integration.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 1. Config load ────────────────────────────────────────────────────────────
from pycwb.config import Config
c = Config()
c.load_from_yaml('user_parameters.yaml')
print(f"[1] Config OK  ifo={c.ifo}  rateANA={c.rateANA}  l_low={c.l_low}  l_high={c.l_high}")

# ── 2. Data source ────────────────────────────────────────────────────────────
from pycwb.modules.online.data_source import SharedMemoryDataSource
src = SharedMemoryDataSource(base_path='/tmp/shm_test', timeout=10, poll_interval=0.1)
src.connect()
assert src.is_alive(), "data source not alive"
print("[2] Data source connected  alive=True")

# ── 3. Read one 60-second chunk ───────────────────────────────────────────────
channels = c.online_channels
chunk = src.read_chunk(channels, start_gps=1257894000, duration=60)
for ch, ts in chunk.items():
    print(f"[3] {ch}: t0={ts.t0}  dur={ts.duration}  rate={ts.sample_rate}  samples={len(ts)}")
src.close()

# ── 4. Build OnlineSegment ────────────────────────────────────────────────────
from pycwb.types.online import OnlineSegment
import time as _time
seg = OnlineSegment(
    index=0,
    ifos=c.ifo,
    segment_gps_start=1257894000,
    segment_gps_end=1257894060,
    seg_edge=c.segEdge,
    sample_rate=c.inRate,
    data_payload=chunk,
    wall_time_received=_time.time(),
    stride=getattr(c, 'online_segment_stride', 8),
    overlap_frac=0.0,
)
print(f"[4] OnlineSegment built  gps={seg.segment_gps_start}-{seg.segment_gps_end}  ifos={seg.ifos}")

# ── 5. Attempt process_online_segment import ──────────────────────────────────
try:
    from pycwb.workflow.subflow.process_online_segment import process_online_segment
    print("[5] process_online_segment imported OK")
except Exception as e:
    print(f"[5] process_online_segment import FAILED: {e}")

print("\n=== All integration checks passed ===")

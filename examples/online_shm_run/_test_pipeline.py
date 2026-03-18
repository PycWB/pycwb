"""
Smoke test: run the full online pipeline for 2 segments then stop.
Run from the online_shm_run directory:
    /path/to/pycwb-dev-py13/python3 _test_pipeline.py
"""
import sys, os, logging, signal, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pipeline-test")

from pycwb.config import Config
from pycwb.modules.online.data_source import SharedMemoryDataSource
from pycwb.types.online import OnlineSegment
from pycwb.workflow.subflow.process_online_segment import process_online_segment

# ── Config ────────────────────────────────────────────────────────────────────
c = Config()
c.load_from_yaml('user_parameters.yaml')
# point to test SHM
c.online_data_source = {'type': 'shm', 'base_path': '/tmp/shm_test',
                        'timeout': 10, 'poll_interval': 0.1}
logger.info("Config loaded: ifo=%s rateANA=%s", c.ifo, c.rateANA)

# ── Data source ───────────────────────────────────────────────────────────────
src = SharedMemoryDataSource(base_path='/tmp/shm_test', timeout=10, poll_interval=0.1)
src.connect()

channels = c.online_channels
seg_dur = int(getattr(c, 'online_segment_duration', 60))
stride  = int(getattr(c, 'online_segment_stride', 8))
gps_start = 1257894000

n_segments = 0
for i in range(2):
    gps = gps_start + i * stride
    gps_end = gps + seg_dur
    if gps_end > gps_start + 120:
        logger.info("Not enough pre-filled data for segment %d, stopping", i)
        break

    logger.info("--- Segment %d: GPS %d - %d ---", i, gps, gps_end)
    t0 = time.time()
    chunk = src.read_chunk(channels, start_gps=gps, duration=seg_dur)
    logger.info("  Data read: %.2f s", time.time() - t0)

    seg = OnlineSegment(
        index=i,
        ifos=c.ifo,
        segment_gps_start=gps,
        segment_gps_end=gps_end,
        seg_edge=c.segEdge,
        sample_rate=c.inRate,
        data_payload=chunk,
        wall_time_received=time.time(),
        stride=stride,
        overlap_frac=0.0,
    )

    t1 = time.time()
    triggers = process_online_segment(c, seg)
    elapsed = time.time() - t1
    logger.info("  process_online_segment returned %d trigger(s) in %.1f s",
                len(triggers), elapsed)
    n_segments += 1

src.close()
logger.info("Pipeline smoke test PASSED (%d segments processed)", n_segments)

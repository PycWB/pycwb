"""
Online search manager — central orchestrator for the live gravitational-wave
search pipeline.

Coordinates data acquisition, parallel segment analysis, trigger handling,
and health monitoring.  Designed to run as a long-lived process launched
from the CLI via ``pycwb online <config.yaml>``.
"""

import json
import logging
import os
import queue
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from pycwb.config import Config
from pycwb.modules.online.data_acquisition import DataAcquisitionManager
from pycwb.modules.online.data_source import create_data_source
from pycwb.modules.online.latency_monitor import LatencyMonitor
from pycwb.modules.online.trigger_handler import TriggerHandler

logger = logging.getLogger(__name__)


def _worker_initializer():
    """Run once per worker process — import heavy modules so they are
    already loaded when ``process_online_segment`` is called."""
    import numpy  # noqa: F401
    try:
        import numba  # noqa: F401
    except ImportError:
        pass


class OnlineSearchManager:
    """Central orchestrator for the online search pipeline.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file (with ``pycwb_schema: extend``
        for online parameters).
    working_dir : str
        Output directory for triggers, catalogs, and state.
    log_level : str
        Python logging level name.
    """

    def __init__(self, config_file: str, working_dir: str = ".",
                 log_level: str = "INFO"):
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.working_dir = os.path.abspath(working_dir)
        os.makedirs(self.working_dir, exist_ok=True)

        # Load config
        logger.info("Loading config from %s", config_file)
        self.config = Config(config_file, working_dir=self.working_dir)

        # Online parameters (from extension schema, with defaults)
        self.segment_duration = float(
            getattr(self.config, "online_segment_duration", 60)
        )
        self.segment_stride = float(
            getattr(self.config, "online_segment_stride", 20)
        )
        self.n_workers = int(
            getattr(self.config, "online_n_workers", 4)
        )
        self.max_queue_depth = int(
            getattr(self.config, "online_max_queue_depth", 8)
        )

        # Shared synchronisation
        self.segment_queue = queue.Queue(maxsize=self.max_queue_depth)
        self.trigger_queue = queue.Queue()  # unbounded
        self.stop_event = threading.Event()

        # Data source
        self.data_source = create_data_source(self.config)

        # Recover last GPS from state file
        initial_gps = self._load_state_gps()

        # Components
        self.data_acq = DataAcquisitionManager(
            config=self.config,
            data_source=self.data_source,
            segment_queue=self.segment_queue,
            stop_event=self.stop_event,
            initial_gps=initial_gps,
        )
        self.trigger_handler = TriggerHandler(
            config=self.config,
            trigger_queue=self.trigger_queue,
            stop_event=self.stop_event,
            working_dir=self.working_dir,
        )
        self.latency_monitor = LatencyMonitor(
            config=self.config,
            segment_queue=self.segment_queue,
            ring_buffers=self.data_acq.ring_buffers,
            stop_event=self.stop_event,
            n_workers=self.n_workers,
        )
        self.executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            max_tasks_per_child=1,
            initializer=_worker_initializer,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start the online search and block until SIGINT/SIGTERM."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "Online search starting — duration=%.0f s, stride=%.0f s, "
            "workers=%d, queue_depth=%d",
            self.segment_duration, self.segment_stride,
            self.n_workers, self.max_queue_depth,
        )

        self.data_acq.start()
        self.trigger_handler.start()
        self.latency_monitor.start()

        pending_futures = {}
        try:
            while not self.stop_event.is_set():
                # Get next segment
                try:
                    seg = self.segment_queue.get(timeout=1)
                except queue.Empty:
                    self._collect_done(pending_futures)
                    continue

                # Submit for analysis
                future = self.executor.submit(
                    _process_segment_entry, self.config, seg,
                )
                pending_futures[future] = seg

                # Collect completed futures (non-blocking)
                self._collect_done(pending_futures)

        except Exception:
            logger.exception("Unexpected error in main loop")
        finally:
            self._shutdown(pending_futures)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_done(self, pending_futures):
        """Harvest completed futures without blocking."""
        done = [f for f in pending_futures if f.done()]
        for f in done:
            seg = pending_futures.pop(f)
            try:
                triggers = f.result()
                latency = time.time() - seg.wall_time_received
                self.latency_monitor.record_latency(latency)
                for t in triggers:
                    self.trigger_queue.put(t)
                self._persist_gps(seg.segment_gps_end)
                logger.info(
                    "Segment %d complete: %d triggers, latency %.1f s",
                    seg.index, len(triggers), latency,
                )
            except Exception:
                logger.exception("Segment %d failed", seg.index)

    def _shutdown(self, pending_futures):
        """Graceful shutdown: drain queues, wait for in-flight work."""
        logger.info("Shutting down online search...")
        self.stop_event.set()
        self.data_acq.stop()

        # Wait for in-flight analysis (with timeout)
        for f in as_completed(pending_futures, timeout=120):
            try:
                triggers = f.result()
                for t in triggers:
                    self.trigger_queue.put(t)
            except Exception:
                pass

        # Signal trigger handler to drain and exit
        self.trigger_queue.put(None)  # sentinel
        self.trigger_handler.join(timeout=30)
        self.latency_monitor.stop()
        self.executor.shutdown(wait=False)
        self.data_source.close()
        logger.info("Online search stopped")

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d, initiating shutdown", signum)
        self.stop_event.set()

    def _persist_gps(self, gps: float):
        """Write last processed GPS to state file for crash recovery."""
        state_path = os.path.join(
            self.working_dir,
            getattr(self.config, "online_state_file", "online_state.json"),
        )
        try:
            with open(state_path, "w") as f:
                json.dump({"last_processed_gps": gps}, f)
        except OSError:
            logger.warning("Failed to persist GPS state to %s", state_path)

    def _load_state_gps(self):
        """Load last processed GPS from state file, or None."""
        state_path = os.path.join(
            self.working_dir,
            getattr(self.config, "online_state_file", "online_state.json"),
        )
        if os.path.isfile(state_path):
            try:
                with open(state_path) as f:
                    data = json.load(f)
                gps = data.get("last_processed_gps")
                if gps is not None:
                    logger.info("Resuming from GPS %.3f (state file)", gps)
                    return float(gps)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read state file %s", state_path)
        return None


def _process_segment_entry(config, online_seg):
    """Top-level function for ProcessPoolExecutor.

    Must be a module-level function (not a method) so it can be pickled.
    """
    from pycwb.workflow.subflow.process_online_segment import (
        process_online_segment,
    )
    return process_online_segment(config, online_seg)

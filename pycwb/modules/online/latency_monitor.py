"""
Latency and health monitoring daemon thread.

Periodically logs queue depth, processing latency, data staleness,
and worker utilisation.  Raises warnings when metrics exceed thresholds.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class LatencyMonitor(threading.Thread):
    """Health monitor for the online search pipeline.

    Parameters
    ----------
    config : Config
        PyCWB configuration.
    segment_queue : queue.Queue
        The segment submission queue (used for depth monitoring).
    ring_buffers : dict
        ``{ifo: RingBuffer}`` — for data staleness checks.
    stop_event : threading.Event
        Shutdown signal.
    n_workers : int
        Total analysis worker count (for utilisation logging).
    """

    def __init__(self, config, segment_queue, ring_buffers,
                 stop_event, n_workers=4):
        super().__init__(daemon=True, name="LatencyMonitor")
        self.config = config
        self.segment_queue = segment_queue
        self.ring_buffers = ring_buffers
        self.stop_event = stop_event
        self.n_workers = n_workers

        self.interval = float(getattr(config, "online_monitor_interval", 5))
        self.latency_threshold = float(
            getattr(config, "online_latency_threshold", 30)
        )
        self.max_queue = int(getattr(config, "online_max_queue_depth", 8))
        self.duration = float(getattr(config, "online_segment_duration", 60))

        self._last_latencies = []
        self._lock = threading.Lock()

    def record_latency(self, latency_seconds: float):
        """Called by the orchestrator after each segment completes."""
        with self._lock:
            self._last_latencies.append(latency_seconds)

    def run(self):
        logger.info("LatencyMonitor started (interval=%.0f s)", self.interval)
        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval)
            if self.stop_event.is_set():
                break
            self._report()
        logger.info("LatencyMonitor stopped")

    def stop(self):
        self.stop_event.set()

    # ------------------------------------------------------------------

    def _report(self):
        depth = self.segment_queue.qsize()
        if depth > self.max_queue // 2:
            logger.warning("Segment queue depth %d exceeds half capacity (%d)",
                           depth, self.max_queue)
        else:
            logger.info("Segment queue depth: %d / %d", depth, self.max_queue)

        # Data staleness per IFO
        # Convert GPS time to Unix time: GPS_epoch = 1980-01-06; offset = 315964800 s
        # lal gives exact leap-second-corrected conversion; fall back to constant offset.
        for ifo, rb in self.ring_buffers.items():
            last = rb.last_gps
            if last is not None:
                try:
                    from lal import gpstime as _gpstime
                    utc_dt = _gpstime.gps_to_utc(last)
                    # Use calendar.timegm to avoid naive-datetime local-tz bug
                    import calendar
                    last_unix = float(calendar.timegm(utc_dt.timetuple()))
                except Exception:
                    last_unix = last + 315964800  # GPS epoch → Unix epoch offset
                staleness = time.time() - last_unix
                if staleness > 2 * self.duration:
                    logger.warning("IFO %s data staleness: %.0f s", ifo, staleness)
                else:
                    logger.debug("IFO %s data staleness: %.1f s", ifo, staleness)

        # Processing latency
        with self._lock:
            recent = list(self._last_latencies)
            self._last_latencies.clear()

        if recent:
            avg = sum(recent) / len(recent)
            mx = max(recent)
            logger.info("Processing latency — avg: %.2f s, max: %.2f s "
                        "(last %d segments)", avg, mx, len(recent))
            if mx > self.latency_threshold:
                logger.warning("Max processing latency %.2f s exceeds "
                               "threshold %.2f s", mx, self.latency_threshold)

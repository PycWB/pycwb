"""
Data acquisition manager — daemon thread that polls a :class:`DataSource`,
fills per-IFO :class:`RingBuffer` instances, and produces
:class:`OnlineSegment` objects on a shared queue.

Implements sliding-window segment production: each segment covers
``duration`` seconds of data, emitted every ``stride`` seconds.
"""

import logging
import queue
import threading
import time

import numpy as np

from pycwb.types.online import OnlineSegment
from pycwb.modules.online.data_source import DataSource
from pycwb.modules.online.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

_MAX_BACKOFF = 60  # seconds


class DataAcquisitionManager(threading.Thread):
    """Daemon thread that acquires data and enqueues analysis segments.

    Parameters
    ----------
    config : Config
        PyCWB configuration (with online extension attributes).
    data_source : DataSource
        Connected data source adapter.
    segment_queue : queue.Queue
        Queue into which :class:`OnlineSegment` objects are placed.
    stop_event : threading.Event
        Set to signal graceful shutdown.
    initial_gps : float or None
        GPS time to start from.  If *None*, starts from current GPS.
    """

    def __init__(self, config, data_source: DataSource,
                 segment_queue: queue.Queue,
                 stop_event: threading.Event,
                 initial_gps: float = None):
        super().__init__(daemon=True, name="DataAcquisitionManager")
        self.config = config
        self.data_source = data_source
        self.segment_queue = segment_queue
        self.stop_event = stop_event

        self.ifos = list(config.ifo)
        self.channels = list(getattr(config, "online_channels", []))
        if not self.channels:
            raise ValueError(
                "online_channels must be set in config "
                "(one channel per IFO, same order as config.ifo)"
            )
        if len(self.channels) != len(self.ifos):
            raise ValueError(
                f"online_channels length ({len(self.channels)}) "
                f"must match config.ifo length ({len(self.ifos)})"
            )

        self.duration = float(getattr(config, "online_segment_duration", 60))
        self.stride = float(getattr(config, "online_segment_stride", 20))
        self.poll_interval = float(getattr(config, "online_poll_interval", 1))
        self.seg_edge = float(config.segEdge)
        self.sample_rate = float(config.inRate)

        # Ring buffers: capacity = duration + stride + 2*segEdge (generous)
        capacity = self.duration + self.stride + 2 * self.seg_edge + 10
        self.ring_buffers = {
            ifo: RingBuffer(capacity, self.sample_rate)
            for ifo in self.ifos
        }

        self._segment_counter = 0
        self._initial_gps = initial_gps

    def run(self):
        """Main acquisition loop."""
        logger.info("DataAcquisitionManager starting (duration=%.0f s, "
                     "stride=%.0f s, poll=%.1f s)",
                     self.duration, self.stride, self.poll_interval)

        self.data_source.connect()

        # Determine starting GPS
        last_gps = self._initial_gps
        if last_gps is None:
            # In real use the first successful read_chunk sets this
            last_gps = self._wait_for_initial_data()
            if last_gps is None:
                logger.error("Could not determine initial GPS. Exiting.")
                return

        # First segment boundary
        next_seg_end = last_gps + self.duration

        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                self._poll_once(last_gps)
                backoff = 1.0  # reset on success

                # Update last_gps from ring buffers
                latest = min(
                    (rb.last_gps for rb in self.ring_buffers.values()
                     if rb.last_gps is not None),
                    default=None,
                )
                if latest is not None:
                    last_gps = latest

                # Check if enough data for the next segment
                if latest is not None and latest >= next_seg_end:
                    self._emit_segment(next_seg_end)
                    next_seg_end += self.stride

            except Exception:
                logger.exception("Data acquisition error, backing off %.1f s", backoff)
                time.sleep(min(backoff, _MAX_BACKOFF))
                backoff = min(backoff * 2, _MAX_BACKOFF)
                continue

            self.stop_event.wait(self.poll_interval)

        logger.info("DataAcquisitionManager stopped")

    def stop(self):
        """Signal the thread to shut down."""
        self.stop_event.set()

    # ------------------------------------------------------------------

    def _wait_for_initial_data(self) -> float:
        """Try to acquire one chunk to determine the initial GPS."""
        for attempt in range(10):
            if self.stop_event.is_set():
                return None
            try:
                data = self.data_source.read_chunk(
                    self.channels, 0, self.poll_interval
                )
                if data:
                    first_ch = next(iter(data.values()))
                    return float(getattr(first_ch, "t0", 0))
            except Exception:
                logger.debug("Waiting for initial data (attempt %d)", attempt + 1)
            time.sleep(self.poll_interval)
        return None

    def _poll_once(self, last_gps: float):
        """Read one poll interval from the data source and append to buffers."""
        data = self.data_source.read_chunk(
            self.channels, last_gps, self.poll_interval,
        )
        for ch, ifo in zip(self.channels, self.ifos):
            ts = data.get(ch)
            if ts is None:
                continue
            arr = np.asarray(ts.data if hasattr(ts, "data") else ts, dtype=np.float64)
            t0 = float(getattr(ts, "t0", last_gps))
            self.ring_buffers[ifo].append(arr, t0)

    def _emit_segment(self, seg_end: float):
        """Build an OnlineSegment and put it on the queue."""
        seg_start = seg_end - self.duration
        padded_start = seg_start - self.seg_edge
        padded_end = seg_end + self.seg_edge

        payloads = []
        for ifo in self.ifos:
            snapshot = self.ring_buffers[ifo].snapshot(padded_start, padded_end)
            payloads.append(snapshot)
            self.ring_buffers[ifo].mark_snapshot()

        overlap_frac = (self.duration - self.stride) / self.duration

        seg = OnlineSegment(
            index=self._segment_counter,
            ifos=self.ifos,
            segment_gps_start=seg_start,
            segment_gps_end=seg_end,
            seg_edge=self.seg_edge,
            sample_rate=self.sample_rate,
            data_payload=payloads,
            wall_time_received=time.time(),
            stride=self.stride,
            overlap_frac=overlap_frac,
        )

        try:
            self.segment_queue.put(seg, timeout=60)
            logger.info("Segment %d emitted [%.1f, %.1f] GPS",
                        self._segment_counter, seg_start, seg_end)
        except queue.Full:
            logger.error("Segment queue full — dropping segment %d",
                         self._segment_counter)

        self._segment_counter += 1

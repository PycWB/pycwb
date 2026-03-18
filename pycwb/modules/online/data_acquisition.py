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

        # Data quality channels — one per IFO (e.g. "H1:DMT-DQ_VECTOR").
        # If empty, DQ checking is skipped and all data is treated as good.
        self.dq_channels = list(getattr(config, "online_dq_channels", []))
        # Bitmask that must be set for data to be analysis-ready.
        # Default = 1 (bit 0 = CBC analysis ready in DMT-DQ_VECTOR).
        self.dq_bits = int(getattr(config, "online_dq_bits", 1))
        # Map IFO name → DQ channel name for quick lookup
        self._dq_map = {ch.split(":")[0]: ch for ch in self.dq_channels}

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

        # First segment boundary — account for seg_edge so the padded window
        # [seg_end - duration - seg_edge, seg_end + seg_edge] starts no
        # earlier than initial_gps (there is no data before that).
        next_seg_end = last_gps + self.duration + self.seg_edge

        logger.info("Entering acquisition loop, initial last_gps=%.3f, "
                    "next_seg_end=%.3f (need %.0f s of data before first "
                    "segment)", last_gps, next_seg_end,
                    self.duration + 2 * self.seg_edge)

        backoff = 1.0
        _last_progress_log = time.time()
        _progress_interval = 10.0  # seconds between INFO progress messages

        # Track the GPS of the last emitted segment's analysis-end boundary.
        # Data in (last_emitted_end, ...] is "unprocessed" for gap logic.
        last_emitted_end = last_gps

        while not self.stop_event.is_set():
            try:
                self._poll_once(last_gps)
                backoff = 1.0  # reset on success

                # --- Check for gaps in any ring buffer ---
                gap_detected = False
                pre_gap_gps = None
                for ifo in self.ifos:
                    detected, pgps = self.ring_buffers[ifo].check_and_clear_gap()
                    if detected:
                        gap_detected = True
                        # Use the minimum pre-gap GPS across all IFOs
                        if pgps is not None:
                            if pre_gap_gps is None or pgps < pre_gap_gps:
                                pre_gap_gps = pgps

                if gap_detected and pre_gap_gps is not None:
                    # Data available before the gap: (last_emitted_end, pre_gap_gps]
                    available_before_gap = pre_gap_gps - last_emitted_end
                    logger.info(
                        "Gap detected at GPS %.3f. Unprocessed data before "
                        "gap: %.1f s (need > %.1f s seg_edge to emit)",
                        pre_gap_gps, available_before_gap, self.seg_edge,
                    )
                    if available_before_gap > 2 * self.seg_edge:
                        # Emit a partial segment covering the pre-gap data.
                        # The analysis window is the available data minus
                        # seg_edge padding on each side.
                        partial_end = pre_gap_gps - self.seg_edge
                        partial_start = max(
                            last_emitted_end,
                            partial_end - self.duration,
                        )
                        partial_dur = partial_end - partial_start
                        if partial_dur > 2 * self.seg_edge:
                            logger.info(
                                "Emitting gap-triggered segment "
                                "[%.1f, %.1f] GPS (%.1f s)",
                                partial_start, partial_end, partial_dur,
                            )
                            self._emit_segment(partial_end,
                                               duration_override=partial_dur)
                            last_emitted_end = partial_end

                    # Reset next_seg_end — after the gap, we need fresh data
                    # so we'll recalculate once new data arrives.
                    next_seg_end = None
                    logger.info(
                        "Segment scheduling reset after gap. Waiting for "
                        "new data to set next segment boundary."
                    )

                # Update last_gps from ring buffers
                latest = min(
                    (rb.last_gps for rb in self.ring_buffers.values()
                     if rb.last_gps is not None),
                    default=None,
                )
                if latest is not None:
                    last_gps = latest

                # After a gap reset, recalculate next_seg_end from new data
                if next_seg_end is None and latest is not None:
                    next_seg_end = latest + self.duration + self.seg_edge
                    last_emitted_end = latest
                    logger.info(
                        "Post-gap: new data at GPS %.3f, next_seg_end "
                        "set to %.3f",
                        latest, next_seg_end,
                    )

                # Check if enough data for the next segment (including
                # seg_edge padding on both sides).
                if latest is not None and next_seg_end is not None:
                    ready_at = next_seg_end + self.seg_edge
                    remaining = ready_at - latest
                    logger.debug(
                        "Ring buffer latest GPS: %.3f, next segment end: "
                        "%.3f, ready at: %.3f (%.1f s until ready)",
                        latest, next_seg_end, ready_at, remaining,
                    )
                    # Periodic INFO progress report
                    now = time.time()
                    if now - _last_progress_log >= _progress_interval:
                        if remaining > 0:
                            logger.info(
                                "Data reading: latest GPS %.3f | "
                                "next segment [%.1f, %.1f] GPS | "
                                "%.1f s until ready",
                                latest,
                                next_seg_end - self.duration,
                                next_seg_end,
                                remaining,
                            )
                        else:
                            logger.info(
                                "Data reading: latest GPS %.3f | "
                                "segment [%.1f, %.1f] GPS ready, emitting",
                                latest,
                                next_seg_end - self.duration,
                                next_seg_end,
                            )
                        _last_progress_log = now
                if (latest is not None and next_seg_end is not None
                        and latest >= next_seg_end + self.seg_edge):
                    # If next_seg_end is so far behind that the ring buffer
                    # no longer covers the required padded window
                    # [seg_end - duration - seg_edge, seg_end + seg_edge],
                    # skip forward to the earliest viable boundary.
                    buf_capacity = (self.duration + self.stride
                                    + 2 * self.seg_edge + 10)
                    earliest_viable = (latest - buf_capacity
                                       + self.duration + self.seg_edge + 1)
                    if next_seg_end < earliest_viable:
                        skipped = (int((earliest_viable - next_seg_end)
                                       / self.stride) + 1) * self.stride
                        next_seg_end += skipped
                        logger.warning(
                            "Skipping %.0f s of segments (insufficient "
                            "buffered data). Next segment end -> %.3f",
                            skipped, next_seg_end,
                        )
                    if latest >= next_seg_end + self.seg_edge:
                        self._emit_segment(next_seg_end)
                        last_emitted_end = next_seg_end
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
        """Return current GPS as the acquisition start time."""
        try:
            from lal import gpstime
            gps = float(gpstime.gps_time_now())
            logger.info("Initial GPS from lal (current time): %.3f", gps)
            return gps
        except ImportError:
            pass

        # Fallback: astropy
        try:
            from astropy.time import Time
            gps = float(Time.now().gps)
            logger.info("Initial GPS from astropy (current time): %.3f", gps)
            return gps
        except ImportError:
            pass

        # Last resort: manual GPS calculation
        import calendar
        GPS_EPOCH = calendar.timegm((1980, 1, 6, 0, 0, 0, 0, 0, 0))
        gps = float(time.time() - GPS_EPOCH + 18)
        logger.info("Initial GPS from system clock: %.3f", gps)
        return gps

    def _poll_once(self, last_gps: float):
        """Read one poll interval from the data source and append to buffers."""
        logger.debug(
            "Polling data source: channels=%s, start_gps=%.3f, duration=%.1f s",
            self.channels, last_gps, self.poll_interval,
        )
        # read_chunk for SHM uses int(start_gps), so ensure we advance to the
        # next integer GPS boundary to avoid re-reading the same second.
        read_start = float(int(last_gps))  # align to integer GPS
        read_dur = max(self.poll_interval, 1.0)

        data = self.data_source.read_chunk(
            self.channels, read_start, read_dur,
        )

        # ── Data quality check ─────────────────────────────────────────────
        # Per-IFO DQ status: True = pass (append to ring buffer)
        dq_ok = {ifo: True for ifo in self.ifos}
        if self.dq_channels:
            dq_data = self.data_source.read_dq_chunk(
                self.dq_channels, read_start, read_dur,
            )
            for ifo in self.ifos:
                dq_ch = self._dq_map.get(ifo)
                if dq_ch is None:
                    continue
                dq_ts = dq_data.get(dq_ch)
                if dq_ts is None:
                    # Channel absent — treat as pass (no DQ info available)
                    logger.debug(
                        "DQ: %s channel not available — treating as pass",
                        ifo,
                    )
                    continue
                arr = np.asarray(
                    dq_ts.data if hasattr(dq_ts, "data") else dq_ts,
                    dtype=np.int64,
                )
                passed = bool(np.all((arr & self.dq_bits) == self.dq_bits))
                dq_ok[ifo] = passed
                if passed:
                    logger.debug(
                        "DQ: %s GPS %.3f pass (bits=0x%x)",
                        ifo, read_start, self.dq_bits,
                    )
                else:
                    logger.warning(
                        "DQ check FAILED for %s at GPS %.3f "
                        "(required bits=0x%x, got %s) — skipping second",
                        ifo, read_start, self.dq_bits, arr.tolist(),
                    )

        # ── Append strain data for IFOs that pass DQ ──────────────────────
        for ch, ifo in zip(self.channels, self.ifos):
            if not dq_ok.get(ifo, True):
                # DQ failed — do not append; gap detection in ring buffer
                # will handle the break in continuity on the next good second
                continue
            ts = data.get(ch)
            if ts is None:
                logger.warning(
                    "No data returned for channel %s at GPS %.3f", ch, last_gps
                )
                continue
            arr = np.asarray(ts.data if hasattr(ts, "data") else ts, dtype=np.float64)
            t0_raw = getattr(ts, "t0", last_gps)
            # gwpy TimeSeries.t0 is an astropy Quantity (seconds); extract float
            t0 = float(t0_raw.value if hasattr(t0_raw, "value") else t0_raw)
            logger.debug(
                "Received %d samples for %s (t0=%.3f, rate=%.0f Hz)",
                len(arr), ifo, t0, self.sample_rate,
            )
            self.ring_buffers[ifo].append(arr, t0)

    def _emit_segment(self, seg_end: float, duration_override: float = None):
        """Build an OnlineSegment and put it on the queue.

        Parameters
        ----------
        seg_end : float
            GPS end of the analysis window (excluding edge padding).
        duration_override : float or None
            If set, use this duration instead of ``self.duration``.
            Used for gap-triggered partial segments.
        """
        from gwpy.timeseries import TimeSeries as GWpyTS

        seg_dur = duration_override if duration_override is not None else self.duration
        seg_start = seg_end - seg_dur
        padded_start = seg_start - self.seg_edge
        padded_end = seg_end + self.seg_edge

        payloads = []
        for ifo, ch in zip(self.ifos, self.channels):
            snapshot = self.ring_buffers[ifo].snapshot(padded_start, padded_end)
            # Wrap as gwpy TimeSeries so downstream from_input() accepts it
            ts = GWpyTS(
                snapshot,
                t0=padded_start,
                sample_rate=self.sample_rate,
                channel=ch,
                name=ch,
            )
            payloads.append(ts)
            self.ring_buffers[ifo].mark_snapshot()

        overlap_frac = (seg_dur - self.stride) / seg_dur if seg_dur > self.stride else 0.0

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

        label = "gap-triggered " if duration_override is not None else ""
        try:
            self.segment_queue.put(seg, timeout=60)
            logger.info("Segment %d %semitted [%.1f, %.1f] GPS (%.1f s)",
                        self._segment_counter, label, seg_start, seg_end,
                        seg_dur)
        except queue.Full:
            logger.error("Segment queue full — dropping segment %d",
                         self._segment_counter)

        self._segment_counter += 1

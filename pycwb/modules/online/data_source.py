"""
Data source adapters for online data acquisition.

Each adapter implements the :class:`DataSource` interface, providing
``connect()``, ``read_chunk()``, ``is_alive()``, and ``close()`` methods.
A factory function :func:`create_data_source` selects the adapter from
the ``online_data_source`` config block.
"""

import logging
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract interface for live data streams."""

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection to the data stream."""

    @abstractmethod
    def read_chunk(self, channels: list, start_gps: float,
                   duration: float) -> Dict[str, object]:
        """Read *duration* seconds of data starting at *start_gps*.

        Returns a dict mapping channel name to a TimeSeries-like object.
        """

    def read_dq_chunk(self, dq_channels: list, start_gps: float,
                      duration: float) -> Dict[str, Optional[object]]:
        """Read data-quality channels; returns ``None`` per channel on failure.

        Unlike :meth:`read_chunk` this method never raises — missing or
        unreadable DQ channels silently return ``None`` so the acquisition
        loop can continue without the DQ check.
        """
        if not dq_channels:
            return {}
        try:
            return self.read_chunk(dq_channels, start_gps, duration)
        except Exception as exc:
            logger.debug("DQ bulk read failed (%s) — treating all as absent", exc)
            return {ch: None for ch in dq_channels}

    @abstractmethod
    def is_alive(self) -> bool:
        """Return ``True`` if the connection is healthy."""

    @abstractmethod
    def close(self) -> None:
        """Release resources and close the connection."""


class NDS2DataSource(DataSource):
    """NDS2 adapter using ``gwpy.timeseries.TimeSeriesDict.get``."""

    def __init__(self, host: str = "", port: int = 0):
        self.host = host
        self.port = port
        self._connected = False

    def connect(self) -> None:
        # gwpy handles the connection internally; we just flag readiness
        self._connected = True
        logger.info("NDS2DataSource ready (host=%s, port=%d)", self.host, self.port)

    def read_chunk(self, channels, start_gps, duration):
        from gwpy.timeseries import TimeSeriesDict

        logger.debug(
            "NDS2: fetching %s [GPS %.3f, %.3f] (%.1f s)",
            channels, start_gps, start_gps + duration, duration,
        )
        kwargs = {}
        if self.host:
            kwargs["host"] = self.host
        if self.port:
            kwargs["port"] = self.port

        data = TimeSeriesDict.get(
            channels,
            start_gps,
            start_gps + duration,
            **kwargs,
        )
        result = {ch: data[ch] for ch in channels}
        for ch, ts in result.items():
            n = len(ts) if hasattr(ts, "__len__") else "?"
            logger.debug("NDS2: received %s: %s samples", ch, n)
        return result

    def is_alive(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False
        logger.info("NDS2DataSource closed")


class KafkaDataSource(DataSource):
    """IGWN Kafka low-latency strain data consumer.

    Consumes 1-second GWF binary blobs from per-IFO Kafka topics.
    A background daemon thread continuously polls the broker and stores
    decoded ``TimeSeries`` frames in a sliding in-memory buffer.
    :meth:`read_chunk` assembles the requested window from the buffer,
    blocking until all required seconds have arrived or *timeout* elapses.

    Parameters
    ----------
    bootstrap_servers : str
        Kafka broker address, e.g. ``"kafka.scimma.org:9092"``.
    topics : dict
        Mapping ``{ifo: topic_name}``, e.g.
        ``{"H1": "igwn.ligo.h1.raw", "L1": "igwn.ligo.l1.raw"}``.
    group_id : str
        Kafka consumer group ID.
    buffer_seconds : int
        How many seconds of data to retain in the in-memory buffer.
    timeout : float
        Seconds to wait for a required frame before raising TimeoutError.
    """

    def __init__(self, bootstrap_servers: str, topics: dict,
                 group_id: str = "pycwb-online",
                 buffer_seconds: int = 300, timeout: float = 60):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.group_id = group_id
        self.buffer_seconds = buffer_seconds
        self.timeout = timeout
        # {channel: {gps_second: TimeSeries}}
        self._buffer: Dict[str, Dict[int, object]] = {}
        self._lock = threading.Lock()
        self._consumer = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> None:
        from confluent_kafka import Consumer
        self._consumer = Consumer({
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })
        self._consumer.subscribe(list(self.topics.values()))
        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop, daemon=True, name="KafkaConsumer"
        )
        self._thread.start()
        logger.info("KafkaDataSource connected to %s, topics %s",
                    self.bootstrap_servers, list(self.topics.values()))

    def read_chunk(self, channels: list, start_gps: float,
                   duration: float) -> Dict[str, object]:
        """Block until all required 1-second frames are buffered, then return."""
        from gwpy.timeseries import TimeSeries
        import numpy as np

        gps_start_int = int(start_gps)
        n_seconds = int(math.ceil(duration))
        required_gps = list(range(gps_start_int, gps_start_int + n_seconds))
        deadline = time.time() + self.timeout
        result = {}

        for ch in channels:
            segments = []
            for gps in required_gps:
                _logged_wait = False
                while True:
                    with self._lock:
                        if gps in self._buffer.get(ch, {}):
                            logger.debug(
                                "Kafka: assembled %s GPS %d from buffer", ch, gps
                            )
                            segments.append(self._buffer[ch][gps])
                            break
                    if not _logged_wait:
                        logger.debug(
                            "Kafka: waiting for %s GPS %d (deadline in %.1f s)",
                            ch, gps, deadline - time.time(),
                        )
                        _logged_wait = True
                    if time.time() > deadline:
                        raise TimeoutError(
                            f"KafkaDataSource: timed out waiting for "
                            f"{ch} GPS {gps}"
                        )
                    time.sleep(0.05)

            if len(segments) == 1:
                result[ch] = segments[0]
            else:
                arr = np.concatenate([s.value for s in segments])
                result[ch] = TimeSeries(
                    arr,
                    t0=segments[0].t0,
                    sample_rate=segments[0].sample_rate,
                    channel=segments[0].channel,
                )
        return result

    def is_alive(self) -> bool:
        return self._running and (
            self._thread is not None and self._thread.is_alive()
        )

    def close(self) -> None:
        self._running = False
        if self._consumer is not None:
            self._consumer.close()
        logger.info("KafkaDataSource closed")

    # ------------------------------------------------------------------

    def _consume_loop(self):
        """Background thread: poll Kafka and decode GWF blobs into the buffer."""
        import io
        from confluent_kafka import KafkaError
        from gwpy.timeseries import TimeSeriesDict

        while self._running:
            msg = self._consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error: %s", msg.error())
                continue
            try:
                ts_dict = TimeSeriesDict.read(io.BytesIO(msg.value()),
                                              format="gwf")
                gps_second = int(
                    float(next(iter(ts_dict.values())).t0.gps)
                )
                with self._lock:
                    for ch, ts in ts_dict.items():
                        self._buffer.setdefault(ch, {})[gps_second] = ts
                        n = len(ts) if hasattr(ts, "__len__") else "?"
                        logger.debug(
                            "Kafka: buffered %s GPS %d (%s samples)",
                            ch, gps_second, n,
                        )
                    # Prune old seconds to cap memory usage
                    cutoff = gps_second - self.buffer_seconds
                    for ch_buf in self._buffer.values():
                        for old in [g for g in list(ch_buf) if g < cutoff]:
                            del ch_buf[old]
            except Exception:
                logger.exception("Failed to decode Kafka message")


class SharedMemoryDataSource(DataSource):
    """Read 1-second GWF files from a low-latency shared-memory directory tree.

    Expects files under ``{base_path}/{ifo}/`` following the standard LIGO
    low-latency naming convention::

        {site}-{ifo}_{stream}-{gps_start}-{duration}.gwf

    Example: ``/dev/shm/kafka/H1/H-H1_llhoft-1257894000-1.gwf``

    The IFO name is extracted from each channel's prefix
    (``"H1:GDS-..."`` → ``"H1"``).  :meth:`read_chunk` reads one file per
    required GPS second, polling until each file appears or *timeout* elapses.

    Parameters
    ----------
    base_path : str
        Root directory containing per-IFO sub-directories,
        e.g. ``"/dev/shm/kafka"``.
    timeout : float
        Seconds to wait for a file to appear before raising TimeoutError.
    poll_interval : float
        Polling interval (seconds) while waiting for new files.
    """

    # Matches e.g. "H-H1_llhoft-1457805590-1.gwf"
    _FILENAME_RE = re.compile(
        r"^[A-Z]-(?P<ifo>[A-Z0-9]+)_(?P<stream>[^-]+)"
        r"-(?P<gps>\d+)-(?P<dur>\d+)\.gwf$"
    )

    def __init__(self, base_path: str = "/dev/shm/kafka",
                 timeout: float = 30, poll_interval: float = 0.1):
        self.base_path = base_path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._connected = False

    def connect(self) -> None:
        if not os.path.isdir(self.base_path):
            logger.warning(
                "SharedMemoryDataSource: base_path %s does not exist yet",
                self.base_path,
            )
        self._connected = True
        logger.info("SharedMemoryDataSource ready (base_path=%s)",
                    self.base_path)

    def read_chunk(self, channels: list, start_gps: float,
                   duration: float) -> Dict[str, object]:
        """Read all required 1-second GWF files, waiting for each to appear."""
        from gwpy.timeseries import TimeSeries
        import numpy as np

        gps_start_int = int(start_gps)
        n_seconds = int(math.ceil(duration))
        required_gps = list(range(gps_start_int, gps_start_int + n_seconds))
        result = {}

        for ch in channels:
            ifo = ch.split(":")[0]   # "H1" from "H1:GDS-..."
            ifo_dir = os.path.join(self.base_path, ifo)
            segments = []
            logger.debug(
                "SHM: reading %s GPS seconds %s from %s",
                ch, required_gps, ifo_dir,
            )

            for gps in required_gps:
                path = self._wait_for_file(ifo_dir, gps)
                logger.debug("SHM: reading %s GPS %d from %s", ch, gps, path)
                # Retry on I/O errors — the generator may still be
                # flushing the file when we first detect it.
                ts = self._read_gwf_with_retry(path, ch)
                n = len(ts) if hasattr(ts, "__len__") else "?"
                logger.debug(
                    "SHM: read %s GPS %d: %s samples (rate=%.0f Hz)",
                    ch, gps, n, float(ts.sample_rate.value
                                      if hasattr(ts.sample_rate, "value")
                                      else ts.sample_rate),
                )
                segments.append(ts)

            if len(segments) == 1:
                result[ch] = segments[0]
            else:
                arr = np.concatenate([s.value for s in segments])
                result[ch] = TimeSeries(
                    arr,
                    t0=segments[0].t0,
                    sample_rate=segments[0].sample_rate,
                    channel=segments[0].channel,
                )
        return result

    def read_dq_chunk(self, dq_channels: list, start_gps: float,
                      duration: float) -> Dict[str, Optional[object]]:
        """Read DQ channels from per-IFO GWF files; returns ``None`` per
        channel when the channel is absent or the file cannot be read."""
        result: Dict[str, Optional[object]] = {}
        for ch in dq_channels:
            try:
                ifo = ch.split(":")[0]
                ifo_dir = os.path.join(self.base_path, ifo)
                gps_int = int(start_gps)
                path = self._find_file(ifo_dir, gps_int)
                if path is None:
                    logger.debug("DQ: no GWF file for %s GPS %d", ch, gps_int)
                    result[ch] = None
                    continue
                result[ch] = self._read_gwf_with_retry(
                    path, ch, optional=True
                )
            except Exception as exc:
                logger.debug("DQ: channel %s unavailable: %s", ch, exc)
                result[ch] = None
        return result

    def is_alive(self) -> bool:
        return self._connected and os.path.isdir(self.base_path)

    def close(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------

    @staticmethod
    def _read_gwf_with_retry(path: str, channel: str,
                             max_retries: int = 5, delay: float = 0.1,
                             optional: bool = False):
        """Read a GWF file, retrying on I/O errors from incomplete writes.

        Parameters
        ----------
        optional : bool
            If ``True``, return ``None`` when the channel is not found in the
            file instead of raising.  Used for DQ channels that may be absent.
        """
        from gwpy.timeseries import TimeSeries
        for attempt in range(max_retries):
            try:
                return TimeSeries.read(path, channel)
            except RuntimeError as exc:
                # Channel not present in file — fail fast when optional
                if optional and "not found" in str(exc).lower():
                    logger.debug("SHM: optional channel %s not in %s",
                                 channel, path)
                    return None
                if attempt == max_retries - 1:
                    if optional:
                        logger.debug(
                            "SHM: giving up on optional channel %s: %s",
                            channel, exc,
                        )
                        return None
                    raise
                logger.debug(
                    "SHM: I/O error reading %s (attempt %d/%d), retrying",
                    path, attempt + 1, max_retries,
                )
                time.sleep(delay)

    def _wait_for_file(self, ifo_dir: str, gps: int) -> str:
        """Block until a GWF file covering *gps* exists; return its path."""
        deadline = time.time() + self.timeout
        _logged_wait = False
        while True:
            path = self._find_file(ifo_dir, gps)
            if path is not None:
                return path
            if not _logged_wait:
                logger.debug(
                    "SHM: waiting for GWF file covering GPS %d in %s "
                    "(timeout %.0f s)",
                    gps, ifo_dir, self.timeout,
                )
                _logged_wait = True
            if time.time() > deadline:
                raise TimeoutError(
                    f"SharedMemoryDataSource: no GWF file covering GPS {gps}"
                    f" in {ifo_dir}"
                )
            time.sleep(self.poll_interval)

    def _find_file(self, ifo_dir: str, gps: int) -> Optional[str]:
        """Return the path of the GWF file in *ifo_dir* covering *gps*,
        or ``None`` if no matching file exists yet."""
        if not os.path.isdir(ifo_dir):
            return None
        for name in os.listdir(ifo_dir):
            m = self._FILENAME_RE.match(name)
            if m is None:
                continue
            file_gps = int(m.group("gps"))
            file_dur = int(m.group("dur"))
            if file_gps <= gps < file_gps + file_dur:
                return os.path.join(ifo_dir, name)
        return None

    def scan_earliest_gps(self, ifos) -> Optional[float]:
        """Scan all IFO sub-directories and return the earliest GPS second
        present across all IFOs, or ``None`` if no files are found yet."""
        earliest = None
        for ifo in ifos:
            ifo_dir = os.path.join(self.base_path, ifo)
            if not os.path.isdir(ifo_dir):
                continue
            for name in os.listdir(ifo_dir):
                m = self._FILENAME_RE.match(name)
                if m is None:
                    continue
                gps = int(m.group("gps"))
                if earliest is None or gps < earliest:
                    earliest = gps
        return float(earliest) if earliest is not None else None


_ADAPTERS = {
    "nds2": NDS2DataSource,
    "kafka": KafkaDataSource,
    "shm": SharedMemoryDataSource,
}


def create_data_source(config) -> DataSource:
    """Factory: create a :class:`DataSource` from config.

    The adapter type is selected from
    ``getattr(config, 'online_data_source', {}).get('type', 'nds2')``.
    """
    ds_cfg = getattr(config, "online_data_source", {}) or {}
    adapter_type = ds_cfg.get("type", "nds2")
    cls = _ADAPTERS.get(adapter_type)
    if cls is None:
        raise ValueError(
            f"Unknown data source type {adapter_type!r}. "
            f"Supported: {list(_ADAPTERS)}"
        )

    if adapter_type == "nds2":
        return NDS2DataSource(
            host=ds_cfg.get("host", ""),
            port=ds_cfg.get("port", 0),
        )
    if adapter_type == "kafka":
        return KafkaDataSource(
            bootstrap_servers=ds_cfg.get("bootstrap_servers", ""),
            topics=ds_cfg.get("topics", {}),
            group_id=ds_cfg.get("group_id", "pycwb-online"),
            buffer_seconds=int(ds_cfg.get("buffer_seconds", 300)),
            timeout=float(ds_cfg.get("timeout", 60)),
        )
    if adapter_type == "shm":
        return SharedMemoryDataSource(
            base_path=ds_cfg.get("base_path", "/dev/shm/kafka"),
            timeout=float(ds_cfg.get("timeout", 30)),
            poll_interval=float(ds_cfg.get("poll_interval", 0.1)),
        )
    raise ValueError(
        f"Unknown data source type {adapter_type!r}. "
        f"Supported: {list(_ADAPTERS)}"
    )

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
        return {ch: data[ch] for ch in channels}

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
                while True:
                    with self._lock:
                        if gps in self._buffer.get(ch, {}):
                            segments.append(self._buffer[ch][gps])
                            break
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

    Example: ``/dev/shm/kafka/H1/H-H1_llhoft-1457805590-1.gwf``

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

            for gps in required_gps:
                path = self._wait_for_file(ifo_dir, gps)
                segments.append(TimeSeries.read(path, ch))

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
        return self._connected and os.path.isdir(self.base_path)

    def close(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------

    def _wait_for_file(self, ifo_dir: str, gps: int) -> str:
        """Block until a GWF file covering *gps* exists; return its path."""
        deadline = time.time() + self.timeout
        while True:
            path = self._find_file(ifo_dir, gps)
            if path is not None:
                return path
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

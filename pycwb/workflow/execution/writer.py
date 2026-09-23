"""Acknowledged, synchronous output transport for supervised workers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing.connection import Connection


import pickle


class OutputClient:
    """Queue-compatible interface with one outstanding message per worker.

    The supervisor acknowledges progress only after preceding products have
    been written. Unlike a background queue feeder, errors reach the caller.
    """

    def __init__(self, connection: Connection, limit: int) -> None:
        self.connection = connection
        self.limit = limit

    def put(self, item: dict[str, Any]) -> None:
        """Send one bounded message and wait for its writer acknowledgement."""
        payload = pickle.dumps(("write", item), protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > self.limit:
            raise ValueError(
                f"Output message exceeds execution.message_limit ({self.limit} bytes)"
            )
        self.connection.send_bytes(payload)
        response = self.connection.recv()
        if response != "ok":
            raise RuntimeError(f"Output writer failed: {response}")


class OutputWriter:
    """One catalog owner. Any write failure is fatal; never acknowledge it."""

    def __init__(self, config: Any, catalog_file: str, buffer_limit: int) -> None:
        self.config = config
        self.catalog_file = catalog_file
        self.buffer_limit = buffer_limit
        self.triggers: list[Any] = []
        self.buffer_bytes = 0

    def flush(self) -> None:
        """Persist buffered triggers before acknowledging corresponding progress."""
        if self.triggers:
            from pycwb.modules.catalog.catalog import Catalog

            Catalog.open(self.catalog_file).add_triggers(self.triggers)
            self.triggers.clear()
            self.buffer_bytes = 0

    def handle(self, item: dict[str, Any], size: int) -> None:
        """Write one message, propagating output or validation failures."""
        kind = item["type"]
        if kind == "trigger":
            if self.buffer_bytes + size > self.buffer_limit:
                self.flush()
            self.triggers.append(item["trigger"])
            self.buffer_bytes += size
            if len(self.triggers) >= 50:
                self.flush()
        elif kind == "wave":
            from pycwb.workflow.subflow.postprocess_and_plots import add_wf_to_wave

            add_wf_to_wave(
                self.config, item["wave_file"], item["event_id"], item["waves"]
            )
        elif kind == "progress":
            from pycwb.modules.catalog.catalog import Catalog

            self.flush()
            Catalog.open(self.catalog_file).add_lag_progress(
                **{k: v for k, v in item.items() if k != "type"}
            )
        else:
            raise ValueError(f"Unknown output message type: {kind!r}")

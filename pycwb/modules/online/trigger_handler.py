"""
Trigger handler — daemon thread that consumes triggers from a queue,
applies deduplication and significance, saves locally, and dispatches
alerts (GraceDB, webhooks).

Errors in alerting are logged but never crash the pipeline.
"""

import logging
import os
import queue
import threading
import time

from pycwb.modules.online.deduplication import TriggerDeduplicator
from pycwb.modules.online.significance import (
    assign_significance,
    load_significance_model,
)

logger = logging.getLogger(__name__)


class TriggerHandler(threading.Thread):
    """Daemon thread that processes triggers from the analysis workers.

    Parameters
    ----------
    config : Config
        PyCWB configuration (with online extension attributes).
    trigger_queue : queue.Queue
        Queue of :class:`OnlineTrigger` objects (or ``None`` sentinel).
    stop_event : threading.Event
        Set to signal graceful shutdown.
    working_dir : str
        Directory for local trigger output.
    """

    def __init__(self, config, trigger_queue: queue.Queue,
                 stop_event: threading.Event,
                 working_dir: str = "."):
        super().__init__(daemon=True, name="TriggerHandler")
        self.config = config
        self.trigger_queue = trigger_queue
        self.stop_event = stop_event
        self.working_dir = working_dir

        # Deduplication
        duration = float(getattr(config, "online_segment_duration", 60))
        stride = float(getattr(config, "online_segment_stride", 20))
        flush_delay = duration + 2 * stride

        self.dedup = TriggerDeduplicator(
            gps_window=float(getattr(config, "online_dedup_window", 0.5)),
            sky_tolerance=float(getattr(config, "online_dedup_sky_tolerance", 5.0)),
            flush_delay=flush_delay,
        )

        # Significance model
        sig_cfg = getattr(config, "online_significance", {}) or {}
        self.model, self.ifar_table = load_significance_model(
            sig_cfg.get("model_path", ""),
            sig_cfg.get("ifar_file", ""),
        )
        self.feature_columns = sig_cfg.get(
            "feature_columns",
            ["rho", "netcc", "penalty", "ecor", "qveto", "qfactor"],
        )

        # Alert config
        self.alert_cfg = getattr(config, "online_alert", {}) or {}

    def run(self):
        logger.info("TriggerHandler started")

        while not self.stop_event.is_set():
            try:
                trigger = self.trigger_queue.get(timeout=5)
            except queue.Empty:
                # Periodic flush of aged triggers
                self._process_finalized(self.dedup._flush())
                continue

            if trigger is None:  # sentinel
                break

            finalized = self.dedup.ingest(trigger)
            self._process_finalized(finalized)

        # Flush remaining on shutdown
        self._process_finalized(self.dedup.flush_all())
        logger.info("TriggerHandler stopped")

    # ------------------------------------------------------------------

    def _process_finalized(self, triggers):
        """Apply significance, save, and alert for each finalized trigger."""
        for trigger in triggers:
            try:
                self._handle_one(trigger)
            except Exception:
                logger.exception("Error handling trigger (segment %d)",
                                 trigger.segment_index)

    def _handle_one(self, trigger):
        """Process a single finalized trigger."""
        # 1. Assign significance
        assign_significance(
            trigger.event, self.model, self.ifar_table,
            self.feature_columns,
        )

        # 2. Save locally
        if self.alert_cfg.get("local_catalog", True):
            self._save_local(trigger)

        # 3. GraceDB upload
        if self.alert_cfg.get("gracedb", False):
            threshold = float(self.alert_cfg.get("gracedb_ifar_threshold", 0.0))
            ifar = getattr(trigger.event, "ifar", 0.0)
            if ifar >= threshold:
                self._upload_gracedb(trigger)

        # 4. Webhook
        webhook_url = self.alert_cfg.get("webhook_url", "")
        if webhook_url:
            self._send_webhook(trigger, webhook_url)

    def _save_local(self, trigger):
        """Save trigger data to disk."""
        from pycwb.modules.workflow_utils.trigger_utils import (
            save_trigger,
            add_event_to_catalog,
        )

        trigger_dir = os.path.join(
            self.working_dir, "triggers",
            f"seg_{trigger.segment_index:06d}",
        )
        os.makedirs(trigger_dir, exist_ok=True)

        trigger_data = (trigger.event, trigger.cluster, trigger.sky_stats)
        try:
            save_trigger(trigger_dir, trigger_data)
            add_event_to_catalog(
                self.working_dir,
                os.path.join(self.working_dir, "catalog"),
                trigger_data,
            )
            logger.info("Saved trigger %s locally", trigger.event.hash_id)
        except Exception:
            logger.exception("Failed to save trigger locally")

    def _upload_gracedb(self, trigger):
        """Upload trigger to GraceDB."""
        try:
            from pycwb.modules.gracedb.gracedb import (
                upload_online_event,
                upload_skymap,
                write_log,
            )

            group = self.alert_cfg.get("gracedb_group", "Burst")
            pipeline = self.alert_cfg.get("gracedb_pipeline", "CWB")
            search = self.alert_cfg.get("gracedb_search", "AllSky")

            graceid = upload_online_event(
                trigger.event,
                group=group,
                pipeline=pipeline,
                search=search,
            )
            if trigger.sky_stats is not None:
                upload_skymap(graceid, trigger.sky_stats)
            write_log(
                graceid,
                f"PyCWB online trigger: segment={trigger.segment_index}, "
                f"GPS={trigger.segment_gps:.3f}, "
                f"IFAR={getattr(trigger.event, 'ifar', 0):.2f} yr",
            )
            logger.info("Uploaded trigger %s to GraceDB as %s",
                        trigger.event.hash_id, graceid)
        except Exception:
            logger.exception("GraceDB upload failed for trigger %s",
                             trigger.event.hash_id)

    def _send_webhook(self, trigger, url):
        """POST trigger as JSON to a webhook URL."""
        try:
            import requests

            timeout = float(self.alert_cfg.get("webhook_timeout", 10))
            payload = {
                "event_id": str(getattr(trigger.event, "hash_id", "")),
                "gps_time": float(getattr(trigger.event, "gps_time",
                                          trigger.segment_gps)),
                "segment_index": trigger.segment_index,
                "ifar": float(getattr(trigger.event, "ifar", 0)),
                "rho": float(getattr(trigger.event, "rho", 0)),
            }
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            logger.info("Webhook sent for trigger %s (status %d)",
                        trigger.event.hash_id, resp.status_code)
        except Exception:
            logger.exception("Webhook failed for trigger %s",
                             trigger.event.hash_id)

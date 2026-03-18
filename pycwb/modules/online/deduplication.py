"""
Sliding-window trigger deduplication.

With overlapping segments (stride < duration), the same astrophysical
signal may be detected in multiple consecutive segments.  This module
buffers recent triggers and keeps only the best-ranked instance of
each physical event before forwarding it downstream.
"""

import logging
import math
import time
from typing import List

from pycwb.types.online import OnlineTrigger

logger = logging.getLogger(__name__)


def _angular_distance_deg(theta1, phi1, theta2, phi2):
    """Great-circle distance between two sky positions (degrees)."""
    t1 = math.radians(theta1)
    p1 = math.radians(phi1)
    t2 = math.radians(theta2)
    p2 = math.radians(phi2)
    cos_d = (math.sin(t1) * math.sin(t2)
             + math.cos(t1) * math.cos(t2) * math.cos(p1 - p2))
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))


class TriggerDeduplicator:
    """Buffer triggers and merge duplicates from overlapping segments.

    Parameters
    ----------
    gps_window : float
        GPS time coincidence window (seconds).
    sky_tolerance : float
        Sky position coincidence tolerance (degrees).
    flush_delay : float or None
        Seconds after ``wall_time_done`` before a trigger is considered
        final.  If *None*, defaults to ``duration + 2 * stride``
        (set externally by the caller).
    """

    def __init__(self, gps_window: float = 0.5,
                 sky_tolerance: float = 5.0,
                 flush_delay: float = None):
        self.gps_window = gps_window
        self.sky_tolerance = sky_tolerance
        self.flush_delay = flush_delay if flush_delay is not None else 100.0
        self.pending: List[OnlineTrigger] = []

    def ingest(self, trigger: OnlineTrigger) -> List[OnlineTrigger]:
        """Add a trigger; return any finalized (flushed) triggers.

        If *trigger* matches a pending trigger within the GPS/sky window,
        the one with the higher ``rho`` (effective correlated SNR, i.e.
        ``rho[0]``) is kept.
        """
        matched = False
        for i, pending_t in enumerate(self.pending):
            if self._is_duplicate(trigger, pending_t):
                # Keep the one with the higher rho (rho[0])
                new_rho = self._get_rho(trigger)
                old_rho = self._get_rho(pending_t)
                if new_rho > old_rho:
                    logger.info(
                        "Dedup: replacing trigger (rho %.4f -> %.4f) "
                        "at GPS %.3f",
                        old_rho, new_rho,
                        getattr(trigger.event, "gps_time",
                                trigger.segment_gps),
                    )
                    self.pending[i] = trigger
                matched = True
                break

        if not matched:
            self.pending.append(trigger)

        return self._flush()

    @staticmethod
    def _get_rho(trigger: OnlineTrigger) -> float:
        """Extract the primary rho value from a trigger's event.

        Works with both legacy ``Event`` (``rho`` is a list) and new
        ``Trigger`` (``rho`` is a scalar).
        """
        rho = getattr(trigger.event, "rho", 0.0)
        if isinstance(rho, (list, tuple)):
            return float(rho[0]) if rho else 0.0
        return float(rho)

    def flush_all(self) -> List[OnlineTrigger]:
        """Flush all pending triggers (e.g. at shutdown)."""
        finalized = list(self.pending)
        self.pending.clear()
        return finalized

    # ------------------------------------------------------------------

    def _is_duplicate(self, a: OnlineTrigger, b: OnlineTrigger) -> bool:
        gps_a = getattr(a.event, "gps_time", a.segment_gps)
        gps_b = getattr(b.event, "gps_time", b.segment_gps)
        if abs(gps_a - gps_b) >= self.gps_window:
            return False

        theta_a = getattr(a.event, "theta", 0.0)
        phi_a = getattr(a.event, "phi", 0.0)
        theta_b = getattr(b.event, "theta", 0.0)
        phi_b = getattr(b.event, "phi", 0.0)
        return _angular_distance_deg(theta_a, phi_a, theta_b, phi_b) < self.sky_tolerance

    def _flush(self) -> List[OnlineTrigger]:
        now = time.time()
        finalized = []
        remaining = []
        for t in self.pending:
            if now - t.wall_time_done > self.flush_delay:
                finalized.append(t)
            else:
                remaining.append(t)
        self.pending = remaining
        if finalized:
            logger.info("Flushed %d deduplicated trigger(s)", len(finalized))
        return finalized

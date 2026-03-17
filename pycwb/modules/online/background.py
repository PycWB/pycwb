"""
Background estimation manager (placeholder).

The initial online release uses pre-trained XGBoost + IFAR files
for significance assignment.  This module reserves the interface for
future automatic background job management (Condor/Slurm batch
submission, result harvesting, and model retraining).
"""

import logging

logger = logging.getLogger(__name__)


class BackgroundManager:
    """Placeholder for background job management."""

    def submit_background_job(self, data_range, config):
        """Submit a batch job to process background lags."""
        raise NotImplementedError(
            "Background job management is not yet implemented. "
            "Use pre-trained XGBoost + IFAR files for significance."
        )

    def check_job_status(self, job_id) -> str:
        """Poll job status. Returns 'running', 'completed', or 'failed'."""
        raise NotImplementedError

    def collect_results(self, job_id):
        """Harvest background triggers from a completed job."""
        raise NotImplementedError

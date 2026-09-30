"""Use native job processing with the MESA whitener selected in the YAML.

The production processor dispatches on ``whiteMethod: mesa`` and handles
injection trials, reconstruction, catalog collection and restart consistently.
This module keeps the example's custom-processor path usable without copying
an obsolete pipeline. Saved NUL waveforms contain reconstruction residuals.
"""

from pycwb.workflow.subflow.process_job_segment_native import process_job_segment

__all__ = ["process_job_segment"]

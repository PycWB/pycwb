"""Waveform reconstruction report — post-production workflow action.

Wraps ``pycwb.modules.reconstruction.waveform_report.generate_reconstruction_report``
as a post-production workflow action so it can be invoked from a YAML
workflow file via ``pycwb post_process``.

Usage in a workflow YAML::

    steps:
      - action: postprocess.waveform_report.generate_waveform_report
        inputs:
          wave_file: ${wave_file}
        args:
          ifo: [L1, H1]
          confidence_level: 0.9
          whitened: true
"""

from __future__ import annotations

import logging
import os

from pycwb.modules.reconstruction.waveform_report import (
    generate_reconstruction_report,
)
from pycwb.post_production.action_spec import action_spec

logger = logging.getLogger(__name__)


@action_spec(
    outputs=["report_folder"],
    inputs=["wave_file"],
    description="Generate aggregate waveform reconstruction quality report",
    display_name="Waveform Reconstruction Report",
    help=(
        "Loads reconstructed and injected waveforms from a wave.h5 file, "
        "computes statistical aggregates (mean, median, confidence intervals) "
        "across the ensemble, and produces diagnostic plots (PNG) and "
        "numerical results (NPZ) in time and frequency domains."
    ),
)
def generate_waveform_report(
    work_dir: str,
    wave_file: str,
    ifo: list[str],
    confidence_level: float = 0.9,
    whitened: bool = False,
    ordering: str = "percentiles",
    plot_median: bool = False,
    plot_mean: bool = False,
    early_start: float = 0.0,
    scale: int = 0,
    max_workers: int = 8,
    **kwargs,
) -> dict:
    """Generate aggregate waveform reconstruction report for one or more IFOs.

    Parameters
    ----------
    work_dir : str
        Base working directory.  Relative paths are resolved against this.
    wave_file : str
        Path to the consolidated ``wave.h5`` file (absolute or relative to
        ``work_dir``).
    ifo : list of str
        Interferometer names to process (e.g. ``["L1", "H1"]``).
    confidence_level : float
        Confidence level for shaded bands (default 0.9 = 90%).
    whitened : bool
        If ``True``, read whitened ``wf_REC_whiten`` / ``wf_INJ_whiten``
        datasets instead of unwhitened ``wf_REC`` / ``wf_INJ``.
    ordering : str
        Bootstrap method for confidence intervals.  One of
        ``"percentiles"``, ``"BCa"``, ``"studentized_bootstrap"``,
        ``"upper"``, ``"lower"``.
    plot_median : bool
        Overlay the median waveform on reconstruction plots.
    plot_mean : bool
        Overlay the mean waveform on reconstruction plots.
    early_start : float
        Extra padding (seconds) before the injection start when slicing
        waveforms.
    scale : int
        Wavelet scale correction factor: each waveform is multiplied by
        ``sqrt(2) ** scale`` before alignment.
    max_workers : int
        Number of parallel workers for waveform synchronization.

    Returns
    -------
    dict
        Keys: ``report_folder`` (output directory), ``ifo`` (list of IFOs
        processed), ``n_processed`` (number of IFOs processed).
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    wave_path = _resolve(wave_file)
    output_folder = os.path.dirname(os.path.abspath(wave_path))

    if not os.path.exists(wave_path):
        raise FileNotFoundError(
            f"Waveform report input file not found: {wave_path}. "
            "Pass a wave.h5 file or a folder containing wave.h5."
        )

    if isinstance(ifo, str):
        ifo = [ifo]

    logger.info(
        "Generating waveform reconstruction report for IFOs %s from %s",
        ifo, wave_path,
    )

    for det in ifo:
        logger.info("Processing %s ...", det)
        generate_reconstruction_report(
            wave_file=wave_path,
            ifo=det,
            output_folder=output_folder,
            confidence_level=confidence_level,
            whitened=whitened,
            ordering=ordering,
            plot_median=plot_median,
            plot_mean=plot_mean,
            early_start=early_start,
            scale=scale,
            max_workers=max_workers,
        )
        logger.info("Finished processing %s", det)

    return {
        "report_folder": os.path.join(output_folder, "reports"),
        "ifo": ifo,
        "n_processed": len(ifo),
    }

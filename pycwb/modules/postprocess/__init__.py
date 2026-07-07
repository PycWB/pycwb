"""
pycwb.modules.postprocess — Post-production analysis suite.

Comprehensive post-processing: XGBoost model training and evaluation,
efficiency curve computation (hrss50 via sigmoid fit), false-alarm rate
calculation, fake open-box studies, waveform reports, and automated
report generation.
"""

from .selection import trigger_selection, filter_real_simulation
from .evaluate import evaluate_efficiency, score_catalog, evaluate_far_rho, score_mdc_catalog
from .far import far_rho_plot, attach_far_and_significance
from .matching import match_simulations
from .job_selector import select_jobs_by_livetime, filter_catalog_by_jobs, compute_livetime
from .train_xgboost import train_xgboost
from .report import standard_background_report
from .report_builder import postproduction_report
from .zero_lag import zero_lag_report
from .fake_openbox import fake_openbox_report
from .waveform_report import generate_waveform_report
from .ranking_metrics import cumulative_event_rate
from .lag_filters import unshifted_job_ids_from_catalog, zero_lag_mask, nonzero_lag_mask
from .random_filter import random_filter_parquet
from .plot_efficiency import compute_hrss50

__all__ = [
    "trigger_selection",
    "filter_real_simulation",
    "evaluate_efficiency",
    "score_catalog",
    "evaluate_far_rho",
    "score_mdc_catalog",
    "far_rho_plot",
    "attach_far_and_significance",
    "match_simulations",
    "select_jobs_by_livetime",
    "filter_catalog_by_jobs",
    "compute_livetime",
    "train_xgboost",
    "standard_background_report",
    "postproduction_report",
    "zero_lag_report",
    "fake_openbox_report",
    "generate_waveform_report",
    "cumulative_event_rate",
    "unshifted_job_ids_from_catalog",
    "zero_lag_mask",
    "nonzero_lag_mask",
    "random_filter_parquet",
    "compute_hrss50",
]

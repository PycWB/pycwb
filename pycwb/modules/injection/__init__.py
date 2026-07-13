"""
pycwb.modules.injection — Simulated GW signal injection.

Generates and schedules simulated GW signal injections into job segments.
Builds injection parameter lists from config, handles sky distribution
sampling, time-of-arrival distributions, and assigns injections to specific
job segments and trials.
"""

from .injection import (
    generate_injection_list_from_config_for_job_segments,
    distribute_inj_in_job_intervals_by_rate,
    distribute_inj_in_job_intervals_by_poisson,
    generate_auxiliary_injection_list_from_config,
    distribute_inj_in_gps_time_by_rate,
    distribute_inj_in_gps_time_by_poisson,
)
from .wf_generator import generate_injection
from .gwsignal_waveform import get_td_waveform
from .strain import generate_strain_from_injection, project_to_detector
from .sky_distribution import generate_sky_distribution
from .par_generator import get_injection_list_from_parameters, hrss_scaling, snr_scaling, repeat

__all__ = [
    "generate_injection_list_from_config_for_job_segments",
    "distribute_inj_in_job_intervals_by_rate",
    "distribute_inj_in_job_intervals_by_poisson",
    "generate_auxiliary_injection_list_from_config",
    "distribute_inj_in_gps_time_by_rate",
    "distribute_inj_in_gps_time_by_poisson",
    "generate_injection",
    "get_td_waveform",
    "generate_strain_from_injection",
    "project_to_detector",
    "generate_sky_distribution",
    "get_injection_list_from_parameters",
    "hrss_scaling",
    "snr_scaling",
    "repeat",
]

"""Construct simulated job-segment data from Gaussian noise and injections."""

import logging

import numpy as np

from pycwb.modules.injection.strain import generate_strain_from_injection
from pycwb.modules.noise.gaussian import generate_noise
from pycwb.types.time_series import TimeSeries as PycwbTimeSeries

logger = logging.getLogger(__name__)

__all__ = [
    "generate_noise_for_job_seg",
    "generate_injections",
    "generate_injection",
]


def generate_noise_for_job_seg(job_seg, sample_rate, f_low=2.0, data=None):
    """Generate Gaussian noise for the padded window of a job segment."""
    logger.info("Generating noise for job segment %s", job_seg.index)
    if "seeds" in job_seg.noise:
        logger.info("Using seeds %s", job_seg.noise["seeds"])
    if "psds" in job_seg.noise:
        logger.info("Using psds %s", job_seg.noise["psds"])
    logger.info("Sample rate: %s", sample_rate)
    logger.info("Low frequency: %s", f_low)

    seeds = job_seg.noise.get("seeds", [None] * len(job_seg.ifos))
    psds = job_seg.noise.get("psds", [None] * len(job_seg.ifos))

    # Generate the full padded window so whitening sees edge samples and the
    # WDM time-frequency map has the same time origin as the real-data path.
    noises = [
        generate_noise(
            psd=psds[i],
            f_low=f_low,
            sample_rate=sample_rate,
            duration=job_seg.padded_duration,
            start_time=job_seg.padded_start,
            seed=seed,
        )
        for i, seed in enumerate(seeds)
    ]

    if data:
        result = [
            noises[i].inject(PycwbTimeSeries.from_input(data[i]))
            for i in range(len(seeds))
        ]
    else:
        result = noises

    logger.info("Generated noise for job segment %s", job_seg.index)
    return result


def generate_injections(config, job_seg, strain=None):
    """Inject all simulations assigned to a job segment into detector data."""
    ifos = job_seg.ifos
    injected = strain

    if not injected:
        injected = [
            PycwbTimeSeries(
                data=np.zeros(int(job_seg.duration * config.inRate)),
                dt=1.0 / config.inRate,
                t0=job_seg.analyze_start,
            )
            for _ in ifos
        ]

    for injection in job_seg.injections:
        injection_strain = generate_strain_from_injection(
            injection, config, injected[0].sample_rate, ifos
        )
        injected = [
            injected[i].inject(injection_strain[i]) for i in range(len(ifos))
        ]

    return injected


# Backward-compatible singular name for job-segment injection orchestration.
generate_injection = generate_injections

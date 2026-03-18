#!/usr/bin/env python3
"""
Fake online data generator for PyCWB pipeline debugging.

Generates 3600 s of coloured Gaussian noise (aLIGOZeroDetHighPower) with a
single CBC injection (taken from tests/sample/user_parameters_injection.yaml)
and writes the data to /dev/shm/kafka/{ifo}/ as 1-second GWF frames that the
SharedMemoryDataSource adapter expects.

File naming convention (llhoft style):
    {site}-{ifo}_llhoft-{GPS}-1.gwf
    e.g.  H-H1_llhoft-1257894000-1.gwf

Usage
-----
    # default: start GPS now, real-time pacing, one injection near centre
    python fake_data_generator.py

    # custom options
    python fake_data_generator.py \\
        --gps-start 1257894000 \\
        --duration  3600 \\
        --sample-rate 16384 \\
        --ifos H1 L1 \\
        --channel GDS-CALIB_STRAIN_CLEAN_C00 \\
        --shm-base /dev/shm/kafka \\
        --realtime          # pace output to wall-clock (1 frame/s)
        --no-injection      # skip the CBC injection

Notes
-----
* The module is self-contained – it only uses pycbc, gwpy, numpy and stdlib.
* Set --no-realtime for fast pre-population of the ring buffer before the
  online pipeline is started.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fake-data] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Noise generation
# ──────────────────────────────────────────────────────────────────────────────

def _make_noise(duration: int, sample_rate: float, f_low: float,
                seed: int, start_time: float):
    """Return a pycbc TimeSeries of coloured Gaussian noise."""
    import pycbc.noise
    import pycbc.psd

    delta_f = 1.0 / 4          # 0.25 Hz resolution – same as mdc.py default
    flen = int(sample_rate / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)

    # Ensure PSD covers full sample-rate bandwidth
    desired = int(1.0 / (1.0 / sample_rate) / psd.delta_f) // 2 + 1
    if len(psd) < desired:
        psd.resize(desired)

    delta_t = 1.0 / sample_rate
    n_samples = int(duration / delta_t)
    noise = pycbc.noise.noise_from_psd(n_samples, delta_t, psd, seed=seed)
    noise._epoch = start_time
    return noise


# ──────────────────────────────────────────────────────────────────────────────
# Injection
# ──────────────────────────────────────────────────────────────────────────────

def _make_injection(gps_time: float, sample_rate: float, ifos: list[str]):
    """
    Generate a GW150914-like CBC injection and return per-IFO pycbc
    TimeSeries signals, projected to each detector.

    Waveform parameters are taken directly from the tests/sample config.
    """
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform

    delta_t = 1.0 / sample_rate
    hp, hc = get_td_waveform(
        approximant="IMRPhenomXPHM",
        mass1=36.0,
        mass2=29.0,
        spin1x=0.3,
        spin1z=0.2,
        spin2z=0.1,
        distance=1030.0,
        inclination=0.0,
        coa_phase=0.0,
        f_lower=16.0,
        delta_t=delta_t,
    )
    hp._epoch += gps_time
    hc._epoch += gps_time

    ra = 0.0
    dec = 0.0
    polarization = 0.0

    signals = []
    for ifo in ifos:
        det = Detector(ifo)
        sig = det.project_wave(hp, hc, ra, dec, polarization)
        signals.append(sig)
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# GWF writer
# ──────────────────────────────────────────────────────────────────────────────

def _site_char(ifo: str) -> str:
    """Return the single-character site prefix, e.g. 'H' for 'H1'."""
    return ifo[0].upper()


def _write_1s_frame(ifo: str, channel: str, gps: int,
                    samples: np.ndarray, sample_rate: float,
                    shm_base: str) -> str:
    """
    Write a 1-second GWF frame for *ifo* to {shm_base}/{ifo}/.

    Returns the written file path.
    """
    from gwpy.timeseries import TimeSeries as GWpyTS

    ifo_dir = os.path.join(shm_base, ifo)
    os.makedirs(ifo_dir, exist_ok=True)

    # llhoft naming convention
    site = _site_char(ifo)
    fname = f"{site}-{ifo}_llhoft-{gps}-1.gwf"
    out_path = os.path.join(ifo_dir, fname)

    ts = GWpyTS(
        samples,
        t0=gps,
        sample_rate=sample_rate,
        channel=f"{ifo}:{channel}",
        name=f"{ifo}:{channel}",
        unit="strain",
    )
    ts.write(out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────

def run(gps_start: int, duration: int, sample_rate: float, ifos: list[str],
        channel: str, shm_base: str, realtime: bool, inject: bool,
        seed_base: int = 42, f_low: float = 16.0,
        max_frames: int | None = None):
    """
    Generate *duration* seconds of fake strain data and write 1-second GWF
    frames to *shm_base*/{ifo}/.

    Parameters
    ----------
    gps_start : int
        GPS start time.
    duration : int
        Total duration in seconds.
    sample_rate : float
        Sample rate in Hz (native; GWF frames are written at this rate).
    ifos : list[str]
        Detector names, e.g. ['H1', 'L1'].
    channel : str
        Channel name suffix (without IFO prefix), e.g. 'GDS-CALIB_STRAIN_CLEAN_C00'.
    shm_base : str
        Root directory, e.g. '/dev/shm/kafka'.
    realtime : bool
        If True, pace output to ~1 frame/s wall-clock.
    inject : bool
        If True, add a single CBC injection near the mid-point of the data.
    seed_base : int
        Base random seed; each IFO gets seed_base + ifo_index.
    f_low : float
        Low-frequency cutoff for noise PSD.
    max_frames : int or None
        If set, stop after writing this many seconds per IFO.
    """
    logger.info(
        "Generating %d s of fake data for %s, GPS start %d",
        duration, ifos, gps_start,
    )

    spf = int(sample_rate)   # samples per frame (1 s)

    # ── 1. Generate full noise buffers ────────────────────────────────────────
    logger.info("Generating coloured noise (%s PSD)…", "aLIGOZeroDetHighPower")
    noise_ts = []
    for i, ifo in enumerate(ifos):
        ts = _make_noise(
            duration=duration,
            sample_rate=sample_rate,
            f_low=f_low,
            seed=seed_base + i,
            start_time=gps_start,
        )
        noise_ts.append(np.array(ts.data))
        logger.info("  %s noise: %d samples @ %.0f Hz", ifo, len(noise_ts[-1]), sample_rate)

    # ── 2. Inject CBC signal ──────────────────────────────────────────────────
    if inject:
        inj_gps = gps_start + duration // 2
        logger.info(
            "Injecting IMRPhenomXPHM (GW150914-like) at GPS %d", inj_gps
        )
        signals = _make_injection(float(inj_gps), sample_rate, ifos)
        for i, sig in enumerate(signals):
            # align signal array with noise buffer
            sig_start_idx = int((float(sig.start_time) - gps_start) * sample_rate)
            sig_end_idx = sig_start_idx + len(sig)
            # clamp to buffer bounds
            buf_start = max(0, sig_start_idx)
            buf_end = min(len(noise_ts[i]), sig_end_idx)
            if buf_end > buf_start:
                src_start = buf_start - sig_start_idx
                src_end = src_start + (buf_end - buf_start)
                noise_ts[i][buf_start:buf_end] += np.array(sig.data)[src_start:src_end]
                logger.info(
                    "  %s injection added: samples [%d, %d)",
                    ifos[i], buf_start, buf_end,
                )

    # ── 3. Write 1-second GWF frames ─────────────────────────────────────────
    n_frames = max_frames if max_frames is not None else duration
    logger.info(
        "Writing %d × 1-second GWF frames to %s…", n_frames, shm_base
    )

    for sec in range(n_frames):
        frame_t0 = time.time()
        gps = gps_start + sec
        s = sec * spf
        e = s + spf

        for i, ifo in enumerate(ifos):
            chunk = noise_ts[i][s:e]
            path = _write_1s_frame(
                ifo=ifo,
                channel=channel,
                gps=gps,
                samples=chunk,
                sample_rate=sample_rate,
                shm_base=shm_base,
            )
            logger.debug("  wrote %s", path)

        if realtime:
            elapsed = time.time() - frame_t0
            sleep_for = max(0.0, 1.0 - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

        if (sec + 1) % 60 == 0:
            logger.info(
                "  … %d / %d frames written (GPS %d)", sec + 1, n_frames, gps
            )

    logger.info("Fake data generation complete.  Wrote %d frames.", n_frames)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gps-start", type=int, default=None,
                   help="GPS start time (default: current GPS time rounded to integer)")
    p.add_argument("--duration", type=int, default=3600,
                   help="Total duration in seconds (default: 3600)")
    p.add_argument("--sample-rate", type=float, default=16384.0,
                   help="Sample rate in Hz (default: 16384)")
    p.add_argument("--ifos", nargs="+", default=["H1", "L1"],
                   help="Detector names (default: H1 L1)")
    p.add_argument("--channel", default="GDS-CALIB_STRAIN_CLEAN_C00",
                   help="Channel name suffix without IFO prefix "
                        "(default: GDS-CALIB_STRAIN_CLEAN_C00)")
    p.add_argument("--shm-base", default="/dev/shm/kafka",
                   help="Base SHM directory (default: /dev/shm/kafka)")
    p.add_argument("--realtime", action="store_true", default=False,
                   help="Pace output to 1 frame/s wall-clock (live simulation)")
    p.add_argument("--no-realtime", dest="realtime", action="store_false",
                   help="Write all frames as fast as possible (default)")
    p.add_argument("--no-injection", dest="inject", action="store_false", default=True,
                   help="Skip the CBC injection")
    p.add_argument("--seed", type=int, default=42,
                   help="Base noise random seed (default: 42)")
    p.add_argument("--f-low", type=float, default=16.0,
                   help="PSD low-frequency cutoff in Hz (default: 16.0)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after writing this many 1-second frames "
                        "(default: write all duration seconds)")
    return p.parse_args()


def _current_gps() -> int:
    """Return the current GPS time as an integer."""
    try:
        from astropy.time import Time
        return int(Time.now().gps)
    except ImportError:
        # fallback: GPS epoch is 1980-01-06 00:00:00 UTC, offset ≈ 18 leap seconds
        import calendar
        GPS_EPOCH = calendar.timegm((1980, 1, 6, 0, 0, 0, 0, 0, 0))
        return int(time.time() - GPS_EPOCH + 18)


if __name__ == "__main__":
    args = _parse_args()

    gps_start = args.gps_start
    if gps_start is None:
        gps_start = _current_gps()
        logger.info("GPS start not specified — using current GPS time: %d", gps_start)

    run(
        gps_start=gps_start,
        duration=args.duration,
        sample_rate=args.sample_rate,
        ifos=args.ifos,
        channel=args.channel,
        shm_base=args.shm_base,
        realtime=args.realtime,
        inject=args.inject,
        seed_base=args.seed,
        f_low=args.f_low,
        max_frames=args.max_frames,
    )

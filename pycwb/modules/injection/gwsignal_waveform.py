"""
GWSignal-based time-domain waveform generator.

Drop-in replacement for ``pycbc.waveform.get_td_waveform`` used as the default
waveform generator in ``pycwb.modules.read_data.mdc``.

The function accepts the same parameter names as pycbc (``mass1``, ``mass2``,
``f_lower``, ``coa_phase``, ``delta_t``, ``approximant``, …) and translates
them to the ``lalsimulation.gwsignal`` conventions before calling
``gwsignal.GenerateTDWaveform``.

Returns a dict ``{'type': 'polarizations', 'hp': TimeSeries, 'hc': TimeSeries}``
suitable for consumption by ``generate_strain_from_injection``.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Parameter name mapping: pycbc key -> GWSignal key
_PYCBC_TO_GWSIGNAL = {
    "coa_phase": "phi_ref",
    "f_lower": "f22_start",
}

# Parameters that are passed through unchanged (same name in both APIs)
_PASSTHROUGH = {
    "mass1", "mass2",
    "spin1x", "spin1y", "spin1z",
    "spin2x", "spin2y", "spin2z",
    "distance", "inclination",
}

# Parameters consumed by the wrapper (not forwarded to GWSignal)
_CONSUMED = {
    "approximant", "delta_t",
    # mdc.py injects these into the kwargs — ignore them
    "gps_time", "ra", "dec", "pol", "t_start", "t_end",
    "generator", "coordsys", "sky_loc",
    "polarization",
}


def get_td_waveform(**kwargs):
    """Generate time-domain polarisations via GWSignal.

    Accepts pycbc-style keyword arguments and returns a dict with
    ``'type': 'polarizations'`` plus ``'hp'`` and ``'hc'`` as
    :class:`pycwb.types.time_series.TimeSeries` objects.
    """
    import lalsimulation.gwsignal as gwsignal
    import astropy.units as u
    from pycwb.types.time_series import TimeSeries

    approximant = kwargs.get("approximant")
    if approximant is None:
        raise ValueError("'approximant' is required in the injection parameters")

    delta_t = kwargs.get("delta_t")
    if delta_t is None:
        raise ValueError("'delta_t' is required in the injection parameters")

    gen = gwsignal.gwsignal_get_waveform_generator(approximant)

    # Build the GWSignal parameter dict with astropy units
    params = {}

    # Mass parameters (solar masses)
    for key in ("mass1", "mass2"):
        if key in kwargs:
            params[key] = float(kwargs[key]) * u.solMass

    # Spin parameters (dimensionless)
    for key in ("spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"):
        if key in kwargs:
            params[key] = float(kwargs[key]) * u.dimensionless_unscaled
        else:
            params[key] = 0.0 * u.dimensionless_unscaled

    # Distance (Mpc)
    if "distance" in kwargs:
        params["distance"] = float(kwargs["distance"]) * u.Mpc

    # Inclination (rad)
    if "inclination" in kwargs:
        params["inclination"] = float(kwargs["inclination"]) * u.rad

    # Phase reference (rad) — pycbc calls it coa_phase, GWSignal calls it phi_ref
    coa_phase = kwargs.get("coa_phase", 0.0)
    params["phi_ref"] = float(coa_phase) * u.rad

    # Frequency parameters (Hz)
    f_lower = kwargs.get("f_lower", 20.0)
    params["f22_start"] = float(f_lower) * u.Hz
    params["f22_ref"] = float(f_lower) * u.Hz

    # Time step
    params["deltaT"] = float(delta_t) * u.s

    # Conditioning flag
    params["condition"] = 1

    # Forward any extra GWSignal-compatible parameters not in our known sets
    for key, val in kwargs.items():
        if key in _CONSUMED or key in _PASSTHROUGH or key in _PYCBC_TO_GWSIGNAL:
            continue
        # Skip None values
        if val is None:
            continue
        # Try to pass through (may fail in GWSignal if unknown)
        logger.debug("Forwarding extra parameter %s=%s to GWSignal", key, val)

    logger.info("Generating waveform via GWSignal (approximant=%s)", approximant)
    hp_gwpy, hc_gwpy = gwsignal.GenerateTDWaveform(params, gen)

    # Convert gwpy TimeSeries to pycwb TimeSeries
    hp = TimeSeries.from_gwpy(hp_gwpy)
    hc = TimeSeries.from_gwpy(hc_gwpy)

    return {
        "type": "polarizations",
        "hp": hp,
        "hc": hc,
    }

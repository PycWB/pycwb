"""Generate simulated detector strain from waveform injection parameters."""

import logging
from warnings import warn

from pycwb.config import Config
from pycwb.types.detector import Detector
from pycwb.types.time_series import TimeSeries as PycwbTimeSeries
from pycwb.utils.module import import_function
from pycwb.utils.skymap_coord import convert_to_celestial_coordinates

logger = logging.getLogger(__name__)

__all__ = ["project_to_detector", "generate_strain_from_injection"]


def project_to_detector(
    hp,
    hc,
    ra,
    dec,
    polarization,
    detectors,
    geocent_end_time,
    ref_ifo="H1",
):
    """Project plus/cross polarizations onto a list of detectors.

    Parameters
    ----------
    hp, hc : TimeSeries
        Plus and cross polarizations in any supported time-series format.
    ra, dec : float
        Right ascension and declination in radians.
    polarization : float
        Polarization angle in radians.
    detectors : list of str
        Detector names, for example ``["H1", "L1"]``.
    geocent_end_time : float
        Geocentric end time added to the epoch of ``hp`` and ``hc``.
    ref_ifo : str, optional
        Unused compatibility argument.

    Returns
    -------
    list of pycwb.types.time_series.TimeSeries
        Projected strain for each detector.
    """
    hp_ts = PycwbTimeSeries.from_input(hp)
    hc_ts = PycwbTimeSeries.from_input(hc)
    hp_ts = PycwbTimeSeries(
        data=hp_ts.data, dt=hp_ts.dt, t0=hp_ts.t0 + geocent_end_time
    )
    hc_ts = PycwbTimeSeries(
        data=hc_ts.data, dt=hc_ts.dt, t0=hc_ts.t0 + geocent_end_time
    )

    return [
        Detector(ifo).project_wave(hp_ts, hc_ts, ra, dec, polarization)
        for ifo in detectors
    ]


def generate_strain_from_injection(
    injection: dict,
    config: Config,
    sample_rate,
    ifos,
) -> list[PycwbTimeSeries]:
    """Generate detector strain from one injection parameter mapping."""
    # PycWB core intentionally does not apply other waveform defaults: doing so
    # could silently overwrite values supplied by a generator or configuration.
    injection["delta_t"] = 1.0 / sample_rate
    logger.info("Generating injection for %s with parameters:\n%s", ifos, injection)

    if "generator" in injection:
        generator = injection["generator"]
    elif "generator" in config.injection:
        generator = config.injection["generator"]
    else:
        warn(
            "No explicit waveform generator specified. Using "
            "'pycwb.modules.injection.gwsignal_waveform.get_td_waveform' as "
            "default. Please specify the generator in the injection parameters.",
            DeprecationWarning,
            stacklevel=2,
        )
        generator = "pycwb.modules.injection.gwsignal_waveform.get_td_waveform"

    if not isinstance(generator, str):
        raise TypeError(
            "Generator should be a string, e.g. "
            "'lalsimulation.gwsignal.generate_td_waveform', "
            f"got {type(generator)}"
        )

    logger.info("Generating waveform using %s", generator)
    generated_data = import_function(generator)(**injection)

    if isinstance(generated_data, tuple):
        warn(
            "Returning hp and hc as tuple is deprecated, please return as dict "
            "with keys hp and hc",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "Returning hp and hc as tuple is going to be deprecated; return "
            "a dict with keys hp and hc"
        )
        generated_data = {
            "type": "polarizations",
            "hp": generated_data[0],
            "hc": generated_data[1],
        }

    if not isinstance(generated_data, dict):
        raise ValueError(
            f"Unsupported return type from waveform generator: {generated_data}; "
            "expected a tuple for hp/hc or a dict for polarizations/strain"
        )

    if "update" in generated_data:
        update = generated_data.pop("update")
        if not isinstance(update, dict):
            raise TypeError(f"update should be a dict, got {type(update)}")
        injection.update(update)

    generated_type = generated_data.pop("type", None)
    if generated_type == "strain":
        logger.info("Strain is generated")
        provided_ifos = set(generated_data)
        if provided_ifos != set(ifos):
            raise ValueError(
                f"Provided ifos {provided_ifos} are not same as the ifos "
                f"{ifos} in config"
            )
        return [PycwbTimeSeries.from_input(generated_data[ifo]) for ifo in ifos]

    if generated_type != "polarizations":
        raise ValueError(
            "Waveform generator dict must set type to 'strain' or "
            "'polarizations'"
        )

    logger.info("Polarizations %s are generated", generated_data.keys())
    if set(generated_data) != {"hp", "hc"}:
        raise NotImplementedError("Only hp and hc polarization is supported for now")

    hp = PycwbTimeSeries.from_input(generated_data["hp"])
    hc = PycwbTimeSeries.from_input(generated_data["hc"])

    gps_end_time = injection.get("gps_time")
    if gps_end_time is None:
        raise ValueError("gps_time is required in the injection parameters")

    right_ascension = injection.get("ra")
    declination = injection.get("dec")
    polarization = injection.get("pol")
    coordinate_system = injection.get("coordsys", "icrs")

    if coordinate_system != "icrs":
        logger.info(
            "Converting from %s to celestial coordinates for injection",
            coordinate_system,
        )
        sky_loc = injection.get("sky_loc")
        if sky_loc is None or len(sky_loc) != 2:
            raise ValueError(
                "sky_loc = [phi,theta] is required in the injection parameters "
                f"when coordinate system is {coordinate_system}, while "
                f"sky_loc: {injection.get('sky_loc')}"
            )
        right_ascension, declination = convert_to_celestial_coordinates(
            sky_loc[0], sky_loc[1], gps_end_time, coordinate_system
        )

    if declination is None or right_ascension is None or polarization is None:
        raise ValueError(
            "ra, dec and pol are required in the injection parameters, while "
            f"ra: {right_ascension}, dec: {declination}, pol: {polarization}"
        )

    logger.info("Projecting polarizations to detectors %s", ifos)
    return project_to_detector(
        hp,
        hc,
        right_ascension,
        declination,
        polarization,
        ifos,
        gps_end_time,
    )

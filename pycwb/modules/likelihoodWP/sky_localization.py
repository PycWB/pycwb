"""Background HEALPix localization using the cWB 2G probability prescription.

The detection sky maximizes the correlation statistic. Localization additionally
uses the final waveform normalization, noise scale and optional antenna prior.
No ROOT objects or global network state are needed here.
"""

from dataclasses import dataclass
import numpy as np
from .sky_order import wave_sort_indices


@dataclass
class SkyLocalization:
    probability: np.ndarray
    indices: np.ndarray
    selected_probability: np.ndarray
    error_regions: np.ndarray


def localize_sky(statistic, antenna_prior, rms, *, use_prior=False, n_sky=0):
    """Return posterior and cWB's sqrt(area) deciles for a full HEALPix grid.

    Nonpositive sky statistics carry zero probability. Invalid scales/maps and
    grids below the release's minimum size return None rather than fabricated
    zero uncertainties. Regions 0 and 10 require an injection/target location;
    they remain zero for ordinary background events, as in the reference.
    """
    stat = np.asarray(statistic, dtype=np.float64)
    antenna = np.asarray(antenna_prior, dtype=np.float64)
    if stat.ndim != 1 or antenna.shape != stat.shape:
        raise ValueError("Sky statistic and antenna prior must be aligned vectors")
    length = len(stat)
    low = length - int(0.9999 * length)
    scale = abs(float(rms))
    valid = np.isfinite(stat) & (stat > 0)
    if low < 2 or not np.isfinite(scale) or scale <= 0 or not np.any(valid):
        return None
    probability = np.zeros(length, dtype=np.float64)
    maximum = np.max(stat[valid])
    probability[valid] = np.exp(-(maximum - stat[valid]) / 2.0 / scale)
    if use_prior:
        maximum_antenna = np.max(antenna)
        if not np.isfinite(antenna).all() or maximum_antenna <= 0 or np.any(antenna < 0):
            return None
        probability[valid] *= (antenna[valid] / maximum_antenna) ** 4
    # cWB accumulates in ascending statistic order, before optional prior sort.
    stat_order = wave_sort_indices(np.where(valid, stat, -np.inf))
    total = 0.0
    for index in stat_order:
        total += probability[index]
    if not np.isfinite(total) or total <= 0:
        return None
    order = wave_sort_indices(probability, stat_order) if use_prior else stat_order
    probability /= total
    ranks = order[:low:-1]  # release loop: l=L-1; l>Lm; --l
    region_counts = np.zeros(11, dtype=np.int64)
    volume = 0.0
    for index in ranks:
        volume += probability[index]
        for decile in range(int(volume * 10) + 1, 10):
            region_counts[decile] += 1
        if volume >= 0.9:
            break
    solid_angle = 4.0 * np.pi * (180.0 / np.pi) ** 2 / length
    regions = np.sqrt(region_counts * solid_angle).astype(np.float32)
    selected = []
    volume = 0.0
    limit = min(int(n_sky), length - low) if n_sky > 0 else int(n_sky)
    threshold = float("0." + str(abs(limit))) if limit < 0 else None
    for index in ranks:
        volume += probability[index]
        if selected and (
            (limit == 0 and (len(selected) == 1000 or volume > 0.99))
            or (limit < 0 and volume > threshold)
            or (limit > 0 and len(selected) == limit)
        ):
            break
        selected.append(index)
    selected = np.asarray(selected, dtype=np.int64)
    return SkyLocalization(probability, selected, probability[selected].astype(np.float32), regions)

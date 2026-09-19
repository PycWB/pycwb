"""Array implementation of the cWB 6.4.6.9 CBC micropixel chirp estimator.

The histogram is local to one accepted event; no pixels are removed from the
cluster. Bootstrap randomness is an explicit input so CPU and future device
implementations can share the same trial sequence.
"""

from dataclasses import dataclass
import math
import numpy as np
from numba import njit


@dataclass(frozen=True)
class ChirpResult:
    """Store release chirp outputs after float32 narrowing.

    Mass is in solar masses; merger time uses the pixel time origin in seconds.
    Inactive fits use zero values. The estimator uses -1 for unavailable mass
    and merger-time errors where required by the release convention.
    """

    mass: float = 0.0
    mass_error: float = 0.0
    merger_time: float = 0.0
    merger_time_error: float = 0.0
    ellipticity: float = 0.0
    energy_fraction: float = 0.0
    symmetry: float = 0.0


def micropixels(pixels, analysis_rate):
    """Return ordered occupied TH2F-equivalent cells and minimum time step.

    Use only the occupied frequency extent, retaining ROOT's underflow row
    and excluding its overflow row. Each overlapping contribution is rounded
    to float32 in original pixel order, as in TH2F::SetBinContent.

    Parameters
    ----------
    pixels : PixelArrays
        Core flags, pixel coordinates, rates, layers and likelihood weights.
    analysis_rate : float
        Analysis sampling rate in Hz.

    Returns
    -------
    cells : numpy.ndarray
        Occupied (time in seconds, frequency in Hz, likelihood) rows.
    minimum_step : float
        Smallest histogram time step in seconds.
    """
    selected = np.flatnonzero(pixels.core)
    if not len(selected):
        return np.empty((0, 3)), 1.0 / analysis_rate
    rate = pixels.rate[selected].astype(float)
    layers = pixels.layers[selected].astype(np.int64)
    if np.any(rate <= 0) or np.any(layers <= 1):
        raise ValueError("Micropixels require positive rates and WDM layers > 1")
    max_layers = int(layers.max())
    min_rate = int(analysis_rate / (max_layers - 1))
    max_rate = int(analysis_rate)
    dt = 1.0 / rate
    starts = (pixels.time[selected] // layers) / rate - dt / 2.0
    lower = float(starts.min()) - 1.0 / min_rate
    upper = float((starts + dt).max()) + 1.0 / min_rate
    n_time = int((upper - lower) * max_rate)
    n_freq = 2 * (max_layers - 1)
    rects = []
    for pos, idx in enumerate(selected):
        m = max_rate // int(rate[pos] + 0.5)
        k = n_freq // (int(layers[pos]) - 1)
        i = int((starts[pos] - lower) * max_rate) + 1
        j = int(pixels.frequency[idx]) * k + 1 - k // 2
        rects.append((i, i + m, max(0, j), min(n_freq + 1, j + k), max(0.0, float(pixels.likelihood[idx]))))
    f0 = min(r[2] for r in rects)
    f1 = max(r[3] for r in rects)
    histogram = np.zeros((n_time + 1, max(0, f1 - f0)), dtype=np.float32)
    for i0, i1, j0, j1, like in rects:
        view = histogram[max(0, i0) : min(n_time + 1, i1), j0 - f0 : j1 - f0]
        # Cast the scalar explicitly: NumPy's weak scalar rule otherwise
        # rounds the input before adding it to the stored float32 value.
        view[:] = view.astype(np.float64) + like
    ti, fi = np.nonzero(histogram > 0)
    dx = (upper - lower) / n_time
    dy = (analysis_rate / 2.0) / n_freq
    return np.column_stack(
        (lower + (ti - 0.5) * dx, (fi + f0 - 0.5) * dy, histogram[ti, fi].astype(float))
    ), 1.0 / max_rate


def root_uniforms(seed, count):
    """TRandom3's one-word MT19937 uniforms for a positive explicit seed.

    Seed zero means time-dependent seeding in ROOT and is intentionally
    unsupported here. Random state is local, including under lag concurrency.

    Parameters
    ----------
    seed : int
        Nonzero uint32 seed matching the run identifier.
    count : int
        Number of nonzero uniform samples to return.

    Returns
    -------
    numpy.ndarray
        Float64 samples in (0, 1), in release generator order.
    """
    if not 0 < int(seed) < 2**32:
        raise ValueError("Chirp bootstrap requires a nonzero uint32 run seed")
    rng = np.random.RandomState(int(seed))
    values = rng.randint(0, 2**32, count, dtype=np.uint32)
    values = values[values != 0]
    while len(values) < count:
        extra = rng.randint(0, 2**32, count - len(values), dtype=np.uint32)
        values = np.concatenate((values, extra[extra != 0]))
    return values.astype(np.float64) * (1.0 / 4294967296.0)


@njit(cache=True)
def _bootstrap(x, f, energy, mindt, uniforms):
    """Run the fixed-order release bootstrap using an explicit uniform stream."""
    n = len(x)
    y = np.empty(n)
    weights = np.empty(n)
    cdf = np.zeros(n + 1)
    sx = sy = 0.0
    for i in range(n):
        y[i] = 1.0 / (f[i] / 128.0) ** (8.0 / 3.0)
        # The oracle takes sqrt(likelihood), squares it in double, then
        # stores the histogram sampling weight in float32.
        weights[i] = math.sqrt(energy[i]) ** 2
        cdf[i + 1] = cdf[i] + np.float32(weights[i])
        sx += x[i]
        sy += y[i]
    cdf /= cdf[n]
    mx, my = sx / n, sy / n
    xx = yy = xy = 0.0
    for i in range(n):
        xx += (x[i] - mx) ** 2
        yy += (y[i] - my) ** 2
        xy += (x[i] - mx) * (y[i] - my)
    delta = math.sqrt((xx - yy) ** 2 + 4 * xy * xy)
    a, b = math.sqrt(max(0.0, (xx + yy + delta) / 2)), math.sqrt(max(0.0, (xx + yy - delta) / 2))
    ellipticity = abs((a - b) / (a + b)) if a + b else 0.0
    used = np.zeros(n, dtype=np.bool_)
    cursor = 0
    best = slope = merger = symmetry = 0.0
    selected = 0
    for trial in range(1000):
        used[:] = False
        sx = sy = sx2 = sxy = 0.0
        for pick in range(6):
            while True:
                if cursor >= len(uniforms):
                    return np.full(5, np.nan), cursor
                r = uniforms[cursor]
                cursor += 1
                cell = np.searchsorted(cdf, r, side="right") - 1
                # TH1F spans [0, n-1] with n bins. Preserve its unusual
                # interpolation and integer truncation before duplicate checks.
                width = (n - 1.0) / n
                sample = cell * width
                if r > cdf[cell]:
                    sample += width * (r - cdf[cell]) / (cdf[cell + 1] - cdf[cell])
                k = int(sample)
                if not used[k]:
                    break
            used[k] = True
            sx += x[k]
            sy += y[k]
            sx2 += x[k] * x[k]
            sxy += x[k] * y[k]
        numerator, denominator = sy * sx - sxy * 6, sx2 * 6 - sx * sx
        if numerator == 0 or denominator == 0:
            continue
        sl = numerator / denominator
        tm = (sy + sl * sx) / 6 / sl
        sl = -sl
        upper = lower = upper_total = lower_total = score = 0.0
        npix = 0
        for i in range(n):
            t, freq, w = x[i] - tm, f[i], weights[i]
            sign = -1.0 if sl < 0 else 1.0
            t1, f1 = t - sign / 64.0, freq - 4.0
            dt1 = t1 - (f1 / 128.0) ** (-8.0 / 3.0) / sl if f1 > 0 else 0.0
            df1 = f1 - 128.0 * (sl * t1) ** (-3.0 / 8.0) if t1 * sign > 0 else 0.0
            if sign * dt1 >= 0 and df1 >= 0:
                upper_total += w
                continue
            t1, f1 = t + sign / 64.0, freq + 4.0
            dt1 = t1 - (f1 / 128.0) ** (-8.0 / 3.0) / sl if f1 > 0 else 0.0
            df1 = f1 - 128.0 * (sl * t1) ** (-3.0 / 8.0) if t1 * sign > 0 else 0.0
            if sign * dt1 <= 0 and df1 <= 0:
                lower_total += w
                continue
            dtime = t - (freq / 128.0) ** (-8.0 / 3.0) / sl if freq > 0 else 0.0
            dfreq = freq - 128.0 * (sl * t) ** (-3.0 / 8.0) if t * sign > 0 else 0.0
            if sign * dtime >= 0 and dfreq >= 0:
                upper += w
                upper_total += w
            if sign * dtime <= 0 and dfreq <= 0:
                lower += w
                lower_total += w
            score += w
            npix += 1
        balance = 1.0 - abs((upper - lower) / (upper + lower)) if upper + lower > 0 else 0.0
        score *= balance
        if score < best:
            continue
        best, slope, merger, selected = score, sl, tm, npix
        symmetry = (
            1.0 - abs((upper_total - lower_total) / (upper_total + lower_total))
            if upper_total + lower_total > 0
            else 0.0
        )
    if selected < 4:
        return np.zeros(5), cursor
    total = signal = 0.0
    for i in range(n):
        total += weights[i]
        offset = 2 * mindt if slope < 0 else -2 * mindt
        t = x[i] - merger - offset
        if t * slope <= 0:
            continue
        df = f[i] - 128.0 * (slope * t) ** (-3.0 / 8.0)
        dt = t - (f[i] / 128.0) ** (-8.0 / 3.0) / slope
        if df > 0:
            if abs(dt) > abs(offset) and abs(df * (dt + offset)) > 1:
                continue
        elif abs(df * dt) > 1:
            continue
        signal += weights[i]
    c = 299792458.0
    mgc = 256.0 * math.pi / 5.0 * (6.67259e-11 * 1.98892e30 * math.pi / c / c / c) ** (5.0 / 3.0) * 128.0 ** (8.0 / 3.0)
    mass = abs(slope / mgc) ** 0.6 * (1 if slope < 0 else -1)
    return np.array([mass, merger, ellipticity, signal / total, symmetry]), cursor


def estimate_chirp(pixels, analysis_rate, seed):
    """Estimate release micropixel chirp morphology with a reproducible bootstrap.

    Parameters
    ----------
    pixels : PixelArrays
        Accepted cluster pixels; the histogram selects core pixels.
    analysis_rate : float
        Analysis sampling rate in Hz.
    seed : int
        Nonzero uint32 run seed for the local uniform stream.

    Returns
    -------
    ChirpResult
        Narrowed release outputs, or the release sentinel values for small clusters.

    Raises
    ------
    ValueError
        If the input seed, pixel geometry or bootstrap sampling is invalid.

    Notes
    -----
    Growing the random stream restarts from the same seed, preserving the prefix
    and reference sampling order. Do not replace this with a global RNG.
    """
    cells, mindt = micropixels(pixels, analysis_rate)
    if len(cells) < 5:
        return ChirpResult()
    if len(cells) < 13:
        return ChirpResult(mass_error=-1.0, merger_time_error=-1.0)
    count = 16384
    while True:
        result, consumed = _bootstrap(cells[:, 0], cells[:, 1], cells[:, 2], mindt, root_uniforms(seed, count))
        if np.all(np.isfinite(result)):
            stored = [float(np.float32(v)) for v in result]
            return ChirpResult(stored[0], -1.0, stored[1], -1.0, *stored[2:])
        if consumed < count:
            raise ValueError("Non-finite chirp estimate")
        count *= 2
        if count > 2**22:
            raise ValueError("Degenerate chirp sampling weights")

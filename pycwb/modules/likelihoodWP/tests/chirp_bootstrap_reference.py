"""Frozen ed13f26 bootstrap oracle: intentionally independent of shared helpers."""
import math
import numpy as np
from numba import njit

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

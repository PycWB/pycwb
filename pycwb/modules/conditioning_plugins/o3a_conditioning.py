"""O3a 16–48 Hz variability correction from the cWB production plugin.

This is the narrow-band post-whitening correction, not PSD_correction.py.
"""
from dataclasses import replace
from .api import HookContext, ConditioningResult
import numpy as np
from numba import njit
from wdm_wavelet.wdm import WDM
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_map import TimeFrequencyMap

HOOK_STAGE = 'post_whitening'
OPTIONS_SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'detectors': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True}}}


@njit(cache=True)
def _split_indices(values, indices, left, right, rank):
    """Reference waveSplit pointer partition, including equal-value ordering."""
    while True:
        i = (right+left)//2
        j = right-1
        if values[indices[left]] > values[indices[i]]:
            indices[left], indices[i] = indices[i], indices[left]
        if values[indices[left]] > values[indices[right]]:
            indices[left], indices[right] = indices[right], indices[left]
        if values[indices[i]] > values[indices[right]]:
            indices[i], indices[right] = indices[right], indices[i]
        if right-left < 3:
            return
        pivot = values[indices[i]]
        indices[i], indices[j] = indices[j], indices[i]
        i = left
        while True:
            i += 1
            while values[indices[i]] < pivot:
                i += 1
            j -= 1
            while values[indices[j]] > pivot:
                j -= 1
            if j < i:
                break
            indices[i], indices[j] = indices[j], indices[i]
        indices[i], indices[right-1] = indices[right-1], indices[i]
        if i == rank:
            return
        if i > rank:
            right = i
        else:
            left = i


@njit(cache=True)
def lpr_filter(data, rate, edge):
    """wavearray::lprFilter(4,0,0,edge,1), including right-tail rejection."""
    size = len(data)
    order = int(rate * 4 + .5)
    count = int(size - 2*edge*rate)
    count -= count & 1
    offset = int(edge*rate+.5)
    offset += offset & 1
    if count <= 2*order+2:
        raise ValueError('Segment too short for O3a LPR estimate')
    begin = (size-count)//2
    x = data[begin:begin+count].copy()
    left = 1
    right = count-int(.06*count+.5)-1
    indices = np.arange(count)
    _split_indices(x, indices, 0, count-1, count//2)
    _split_indices(x, indices, 0, count//2, left)
    _split_indices(x, indices, count//2, count-1, right)
    median = x[indices[count//2]]
    x -= median
    for i in range(left):
        x[indices[i]] = 0.
    for i in range(right, count):
        x[indices[i]] = 0.
    r = np.zeros(order)
    for m in range(order):
        for i in range(order, count-order):
            r[m] += x[i]*(x[i-m]+x[i+m])/2.
    # Constant envelopes have no predictable variability. Reference divides by zero;
    # the explicit identity fallback is safe and is recorded by the caller.
    if r[0] <= 0:
        return data.copy()
    a = np.zeros(order)
    b = np.zeros(order)
    a[0] = 1.
    s = r[0]
    for m in range(1, order):
        q = 0.
        for i in range(m):
            q -= a[i]*r[m-i]
        if s <= 0 or not np.isfinite(s):
            raise ValueError('Degenerate O3a LPR covariance')
        reflection = q/s
        s -= q*reflection
        for i in range(1, m+1):
            b[i] = a[i]+reflection*a[m-i]
        for i in range(1, m+1):
            a[i] = b[i]
    out = data.copy()
    n = offset+order
    for i in range(n, size):
        weight = .5 if i < size-n else 1.
        for j in range(1, order):
            out[i] += a[j]*data[i-j]*weight
    for i in range(size-n):
        weight = .5 if i >= n else 1.
        for j in range(1, order):
            out[i] += a[j]*data[i+j]*weight
    return out


def _reference_median(x, edge, rate):
    # wavearray::median has inclusive bounds and an upper order statistic.
    low = int(edge*rate)
    high = int((len(x)/rate-edge)*rate)
    values = x[low:high+1]
    k = len(values)//2 + (len(values)&1)
    return float(np.partition(values, k)[k])


@njit(cache=True)
def _multiplicity(u, cleaned, um, vm, rate, edge):
    size = len(u)
    v = np.empty(size)
    w = np.ones(size)
    for i in range(size):
        value = max(0., cleaned[i]+um-vm)
        vr = u[i]-value
        aa = (u[i]-um)/um if u[i] < 2*um else 1.
        if vr < 0 or aa < 0:
            aa = 0.
        vr *= vr if vr < 1 else 1.
        v[i] = u[i]-vr*aa
        w[i] = v[i]/u[i] if u[i] > 0 else 1.
    n = int(edge*rate)
    m = int(.6*rate)
    p = 1-w
    p[p > .9] = 0.
    q = np.zeros(m)
    for i in range(n+m, size-n-m):
        for j in range(m):
            q[j] += p[i]*(p[i-j]+p[i+j])
    variation = np.ones(size)
    # No correction activity: avoid the reference's zero autocorrelation divisor.
    if q[0] > 0:
        q /= q[0]
    for i in range(m+1, size-m-1):
        sp = sm = 0.
        for j in range(8, m):
            sm += p[i-j]*p[i-j]*q[j]
            sp += p[i+j]*p[i+j]*q[j]
        aa = 1./(1.+max(sm,sp))
        v[i] *= aa
        w[i] *= aa
        variation[i] = w[i] if w[i] != 0 else 1.e-16
    return v, variation


def correct_layer(layer: np.ndarray, rate: float, edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Return corrected amplitudes and variation; sample rate is Hz and edge is seconds."""
    u = np.abs(layer)
    um = _reference_median(u, edge, rate)
    if um <= 0:
        return layer.copy(), np.ones(len(u))
    cleaned = lpr_filter(u, rate, edge)
    vm = _reference_median(cleaned, edge, rate)
    v, variation = _multiplicity(u, cleaned, um, vm, rate, edge)
    corrected = layer * (v / np.where(u > 0, u, 1.))
    return corrected, variation


def correct_strain(strain: TimeSeries, edge: float) -> tuple[TimeSeries, TimeFrequencyMap]:
    """Correct the 16–48 Hz band; reject rates below 128 Hz and preserve input strain."""
    ts = TimeSeries.from_input(strain)
    rate = float(ts.sample_rate)
    layers = int(rate/64+.1)
    if layers < 2:
        raise ValueError('O3a conditioning requires a sample rate of at least 128 Hz')
    wdm = WDM(M=layers, K=layers, beta_order=6, precision=10)
    tf = wdm.t2w(np.asarray(ts.data, dtype=np.float64), sample_rate=rate, t0=float(ts.t0), MM=-1)
    layer_rate = 1./float(tf.dt)
    corrected, variation = correct_layer(np.asarray(tf.data[1]), layer_rate, edge)
    tf.data[1] = corrected
    data = .5*(np.asarray(wdm.w2t(tf))+np.asarray(wdm.w2tQ(tf)))
    # cWB nVAR is a WSeries<float> time series with an affected frequency band,
    # not another wavelet transform. Store it as one band in the normal TF map.
    variation_map = TimeFrequencyMap(
        data=variation.astype(np.float32)[None, :], is_whitened=False,
        dt=float(tf.dt), df=32., start=float(ts.t0),
        stop=float(ts.t0) + len(variation) * float(tf.dt),
        f_low=16., f_high=48., edge=float(edge), wavelet=None,
    )
    return TimeSeries(data=data, t0=ts.t0, dt=ts.dt), variation_map


def apply(context: HookContext, result: ConditioningResult, detectors: list[str] | None = None) -> ConditioningResult:
    """Replace selected detector products in place; reject stacked variation corrections."""
    selected = set(context.ifos).intersection({"L1", "H1"}) if detectors is None else set(detectors)
    if not selected.issubset(context.ifos):
        raise ValueError('O3a detector selection contains an unknown detector')
    for i, ifo in enumerate(context.ifos):
        if ifo not in selected:
            continue
        if getattr(result.noise_rms[i], 'variation', None) is not None:
            raise ValueError('Multiple noise-variation corrections require an explicit composition rule')
        strain, variation = correct_strain(result.strains[i], float(context.config.segEdge))
        result.strains[i] = strain
        result.noise_rms[i] = replace(result.noise_rms[i], variation=variation)
        result.diagnostics.append(dict(kind='o3a_conditioning', detector=ifo,
                                       variation_min=float(np.min(variation.data)),
                                       variation_max=float(np.max(variation.data))))
    return result

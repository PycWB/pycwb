"""cWB pixel-selection gate (not a strain-zeroing operation)."""
import numpy as np
from numba import njit
from .api import ConditioningResult
from pycwb.types.time_series import TimeSeries

HOOK_STAGE = 'time_vetoes'
OPTIONS_SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'energy_threshold': {'type': 'number', 'exclusiveMinimum': 0},
    'integration_seconds': {'type': 'number', 'exclusiveMinimum': 0},
    'padding_seconds': {'type': 'number', 'minimum': 0}}}


@njit(cache=True)
def gate_mask(data, rate, edge, threshold, integration, padding):
    size = len(data)
    window = max(1, int(integration * rate))
    if window > size:
        raise ValueError('Gate integration window exceeds segment')
    margin = int(padding * rate) + 1
    scratch = int(edge * rate + 0.5)
    # Difference array avoids O(glitch duration * padding) marking work.
    changes = np.zeros(size + 1, dtype=np.int64)
    energy = 0.
    for j in range(window):
        energy += data[window - 1 - j] ** 2
    for i in range(window - 1, size):
        if i >= window:
            energy -= data[i-window] ** 2
            energy += data[i] ** 2
        if energy > threshold:
            a = max(i-margin, scratch)
            b = min(i+margin, size-scratch)
            if b > a:
                changes[a] += 1
                changes[b] -= 1
    mask = np.zeros(size, dtype=np.int8)
    active = 0
    for i in range(size):
        active += changes[i]
        mask[i] = active > 0
    mask[0] = mask[-1] = 0
    return mask


def gate_intervals(strain, edge, energy_threshold=1.e6, integration_seconds=.5, padding_seconds=1.5):
    ts = TimeSeries.from_input(strain)
    rate = float(ts.sample_rate)
    mask = gate_mask(np.asarray(ts.data, dtype=np.float64), rate, edge,
                     energy_threshold, integration_seconds, padding_seconds)
    derivative = np.diff(mask)
    starts = np.flatnonzero(derivative == 1)
    stops = np.flatnonzero(derivative == -1)
    # Deliberately reproduce the reference's integer casts, not floor/ceil replacements.
    return [(float(ts.t0) + int(a/rate), float(ts.t0) + int(b/rate+.5))
            for a, b in zip(starts, stops) if int(b/rate+.5) > int(a/rate)]


def apply(context, result, **options):
    from pycwb.types.job import _merge_intervals
    per_detector = {}
    for ifo, strain in zip(context.ifos, result.strains):
        per_detector[ifo] = gate_intervals(strain, float(context.config.segEdge), **options)
    union = _merge_intervals([interval for intervals in per_detector.values() for interval in intervals])
    result.excluded_intervals = _merge_intervals(result.excluded_intervals + union)
    result.diagnostics.append(dict(kind='cwb_gating', per_detector=per_detector,
                                   gated_seconds=sum(b-a for a,b in union)))
    return result

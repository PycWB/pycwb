"""Typed, ordered hooks for native conditioning and selection.

Plugins export ``apply(context, result, **options)`` and
``OPTIONS_SCHEMA``. Hooks run once per trial, before lag-dependent work.
"""
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import hashlib
import json

import jsonschema
import numpy as np

from pycwb.types.time_series import TimeSeries


@dataclass(frozen=True)
class HookContext:
    config: object
    ifos: tuple[str, ...]
    segment: object


@dataclass
class ConditioningResult:
    strains: list
    noise_rms: list
    excluded_intervals: list[tuple[float, float]] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)


def run_hooks(config, segment, strains, noise_rms):
    """Execute validated hooks; empty configuration preserves inputs exactly."""
    result = ConditioningResult(list(strains), list(noise_rms))
    context = HookContext(config, tuple(segment.ifos), segment)
    stages = (
        (getattr(config, 'conditioning', {}) or {}).get('post_whitening', []),
        (getattr(config, 'selection', {}) or {}).get('time_vetoes', []),
    )
    for stage, specs in zip(('post_whitening', 'time_vetoes'), stages):
        for spec in specs:
            module = import_module(spec['module'])
            if getattr(module, 'HOOK_STAGE', None) != stage:
                raise ValueError(f"{spec['module']} does not implement {stage}")
            options = spec.get('options', {})
            jsonschema.validate(options, module.OPTIONS_SCHEMA)
            old = [TimeSeries.from_input(x) for x in result.strains]
            result = module.apply(context, result, **options)
            if not isinstance(result, ConditioningResult):
                raise TypeError('Conditioning plugin must return ConditioningResult')
            if len(result.strains) != len(old) or len(result.noise_rms) != len(old):
                raise ValueError('Plugin changed detector count')
            for original, new in zip(old, result.strains):
                new = TimeSeries.from_input(new)
                if (len(new.data), new.t0, new.dt) != (len(original.data), original.t0, original.dt):
                    raise ValueError('Plugin changed strain timeline')
                if not np.isfinite(new.data).all():
                    raise ValueError('Plugin produced nonfinite strain')
            path = Path(module.__file__)
            result.diagnostics.append(dict(stage=stage, module=spec['module'], options=options,
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def subtract_intervals(keep, excluded, start, stop):
    """Subtract exclusions from accepted intervals; None means the full span."""
    output = [(start, stop)] if keep is None else list(keep)
    for low, high in sorted(excluded):
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError('Invalid plugin exclusion interval')
        next_output = []
        for a, b in output:
            if b <= low or a >= high:
                next_output.append((a, b))
            else:
                if a < low:
                    next_output.append((a, low))
                if b > high:
                    next_output.append((high, b))
        output = next_output
    return output


def save_diagnostics(result, directory):
    """Save one independent directory per job/trial, including variation arrays."""
    if not result.diagnostics:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, noise in enumerate(result.noise_rms):
        variation = getattr(noise, 'variation', None)
        if variation is not None:
            # Preserve the existing one-dimensional diagnostic file format.
            arrays[f'nvar_{i}'] = np.asarray(variation.data[0], dtype=np.float64)
    if arrays:
        np.savez_compressed(path / 'noise_variation.npz', **arrays)
    metadata = dict(plugins=result.diagnostics, excluded_intervals=result.excluded_intervals,
                    noise_variation=[None if getattr(n, 'variation', None) is None else
                                     dict(start=n.variation.start, rate=1.0 / n.variation.dt,
                                          low=n.variation.f_low, high=n.variation.f_high)
                                     for n in result.noise_rms])
    (path / 'diagnostics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))

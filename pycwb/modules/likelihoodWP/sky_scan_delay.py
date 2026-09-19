"""Sky scan grouped by identical detector delays; per-direction math is unchanged.

The group order is prepared once for a sky grid. Only data loads, pixel energy
and the initial mask are shared within a group. DPF and every sky statistic
are recomputed for each valid direction. Final tie-breaking uses the original
sky_valid_indices order, independent of group traversal order.
"""
from math import sqrt
import numpy as np
from numba import njit, prange, float32
from .dpf import dpf_np_loops_vec
# The shared reduced-correlation kernel retains the release LL + 0.001 offset.
from .sky_stat import avx_GW_ps, avx_ort_ps, avx_stat_ps, load_data_from_td


def make_delay_groups(ml):
    """Return all sky indices grouped by equal integer detector-delay tuples."""
    _, inverse = np.unique(ml.T, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    counts = np.bincount(inverse)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return order, offsets


def delay_groups_for_grid(setup, ml, big_cluster=False):
    """Cache immutable setup geometry separately for main and coarse grids.

    Per-cluster sky masks are deliberately not cached here. As with the other
    prepared sky geometry, callers must replace/rebuild ml rather than mutate
    its contents in place after preparing a setup.
    """
    cache = setup.setdefault("_delay_group_cache", {})
    key = "big" if big_cluster else "main"
    cached = cache.get(key)
    if cached is None or cached[0] is not ml:
        cached = (ml, *make_delay_groups(ml))
        cache[key] = cached
    return cached[1:]


@njit(cache=True, parallel=True)
def scan_sky_grouped_delays(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC, delta_regulator, network_energy_threshold, sky_valid_indices, group_order, group_offsets):
    # Keep legacy parameter names for keyword compatibility, but use readable
    # local names inside the scan.
    plus_antenna_patterns = FP
    cross_antenna_patterns = FX
    noise_weights = rms
    td_phase0 = td00
    td_phase90 = td90
    sky_delay_samples = ml

    # Arrays are pre-transposed and cast to float32 by setup_likelihood / the caller.
    regularization_arr = REG.astype(np.float32)

    # --- Allocate per-sky-location statistics arrays ---
    alignment_by_sky = np.zeros(n_sky, dtype=float32)
    likelihood_by_sky = np.zeros(n_sky, dtype=float32)
    null_energy_by_sky = np.zeros(n_sky, dtype=float32)
    coherent_energy_by_sky = np.zeros(n_sky, dtype=float32)
    correlation_by_sky = np.zeros(n_sky, dtype=float32)
    sky_stat_by_sky = np.zeros(n_sky, dtype=float32)
    probability_by_sky = np.zeros(n_sky, dtype=float32)
    disbalance_by_sky = np.zeros(n_sky, dtype=float32)
    network_index_by_sky = np.zeros(n_sky, dtype=float32)
    ellipticity_by_sky = np.zeros(n_sky, dtype=float32)
    polarisation_by_sky = np.zeros(n_sky, dtype=float32)
    antenna_prior_by_sky = np.zeros(n_sky, dtype=float32)

    offset = int(td_phase0.shape[0] / 2)
    # best_stat_by_sky is initialised to -1e12 so that masked / netCC-rejected directions
    # never win the tie-breaking scan (mirrors C++ skyProb.data[l] = -1.e12).
    best_stat_by_sky = np.full(n_sky, np.float32(-1.e12))
    n_valid = len(sky_valid_indices)
    valid = np.zeros(n_sky, dtype=np.bool_)
    for k in range(n_valid):
        valid[sky_valid_indices[k]] = True
    for group in prange(len(group_offsets) - 1):
        begin, end = group_offsets[group], group_offsets[group + 1]
        first = -1
        for position in range(begin, end):
            candidate = group_order[position]
            if valid[candidate]:
                first = candidate
                break
        if first < 0:
            continue
        sky_idx = first
        # --- Apply time delay and load pixel data for this sky direction ---
        data_phase0 = np.empty((n_ifo, n_pix), dtype=float32)
        data_phase90 = np.empty((n_ifo, n_pix), dtype=float32)
        for i in range(n_ifo):
            data_phase0[i] = td_phase0[sky_delay_samples[i, sky_idx] + offset, i]
            data_phase90[i] = td_phase90[sky_delay_samples[i, sky_idx] + offset, i]

        # --- Compute data energy and pixel mask ---
        total_data_energy, _, energy_total, input_mask = load_data_from_td(data_phase0, data_phase90, network_energy_threshold)

        for position in range(begin, end):
            sky_idx = group_order[position]
            if not valid[sky_idx]:
                continue
            mask = input_mask
            # --- Compute DPF (dominant polarisation frame) f+/fx and their norms ---
            _, dominant_plus, dominant_cross, plus_norm, cross_norm, rotation_sin, rotation_cos, network_index = (
                dpf_np_loops_vec(plus_antenna_patterns[sky_idx], cross_antenna_patterns[sky_idx], noise_weights)
            )

            # --- Project data onto GW strain packet; select pixels above threshold ---
            active_pixel_count, signal_phase0, signal_phase90, mask, _, _, _, _ = avx_GW_ps(
                data_phase0, data_phase90,
                dominant_plus, dominant_cross,
                plus_norm, cross_norm, network_index,
                energy_total, mask, regularization_arr,
            )

            # --- Orthogonalise signal amplitudes (+ and x polarisations) ---
            _, rotation_sin, rotation_cos, energy_plus, energy_cross = avx_ort_ps(signal_phase0, signal_phase90, mask)

            # --- Compute coherent network statistics ---
            ellipticity, coherent_energy, polarisation, null_energy, _, _, _ = avx_stat_ps(
                data_phase0, data_phase90, signal_phase0, signal_phase90,
                rotation_sin, rotation_cos, mask,
            )

            disbalance = null_energy / (n_ifo * active_pixel_count + sqrt(active_pixel_count))
            noise_correction = disbalance if disbalance > float(1.0) else 1.0
            correlation = coherent_energy / (coherent_energy + null_energy * noise_correction - active_pixel_count * (n_ifo - 1))

            if not np.isfinite(ellipticity) or ellipticity < netCC:
                continue

            # --- Sky statistics: likelihood and cross-correlation ---
            likelihood_stat = total_data_energy - null_energy if total_data_energy > float32(0.) else float32(0.)
            cross_correlation_stat = likelihood_stat * correlation
            probability_by_sky[sky_idx] = likelihood_stat if delta_regulator < 0 else cross_correlation_stat

            # --- Antenna sensitivity: energy-weighted f+/fx norms ---
            plus_weighted_energy = float32(0.)
            cross_weighted_energy = float32(0.)
            selected_data_energy = float32(0.)

            for j in range(n_pix):
                if mask[j] <= 0:
                    continue
                selected_data_energy += energy_total[j]
                plus_weighted_energy += plus_norm[j] * energy_total[j]
                cross_weighted_energy += cross_norm[j] * energy_total[j]
            plus_weighted_energy = (
                plus_weighted_energy / selected_data_energy
                if selected_data_energy > float32(0.) else float32(0.)
            )
            cross_weighted_energy = (
                cross_weighted_energy / selected_data_energy
                if selected_data_energy > float32(0.) else float32(0.)
            )

            antenna_prior_by_sky[sky_idx] = sqrt(plus_weighted_energy + cross_weighted_energy)
            alignment_by_sky[sky_idx] = (
                sqrt(cross_weighted_energy / plus_weighted_energy)
                if plus_weighted_energy > float32(0.) else float32(0.)
            )
            # --- Store all per-sky statistics ---
            likelihood_by_sky[sky_idx] = total_data_energy - null_energy
            null_energy_by_sky[sky_idx] = null_energy
            coherent_energy_by_sky[sky_idx] = coherent_energy
            correlation_by_sky[sky_idx] = correlation
            sky_stat_by_sky[sky_idx] = cross_correlation_stat
            disbalance_by_sky[sky_idx] = disbalance
            network_index_by_sky[sky_idx] = noise_correction
            ellipticity_by_sky[sky_idx] = ellipticity
            polarisation_by_sky[sky_idx] = polarisation

            best_stat_by_sky[sky_idx] = cross_correlation_stat
    # Mirror C++ tie-breaking: C++ uses `if (AA >= STAT)` in a forward loop,
    # so the LAST pixel with the maximum value wins on ties.
    # Iterate only over valid (unmasked) indices to match C++ behaviour.
    sky_stat_max = 0
    l_max = int(sky_valid_indices[0])
    for _k in range(n_valid):
        _l = sky_valid_indices[_k]
        if best_stat_by_sky[_l] >= sky_stat_max:
            sky_stat_max = best_stat_by_sky[_l]
            l_max = _l

    return (l_max, antenna_prior_by_sky, alignment_by_sky, likelihood_by_sky, null_energy_by_sky, coherent_energy_by_sky, \
              correlation_by_sky, sky_stat_by_sky, disbalance_by_sky, network_index_by_sky, ellipticity_by_sky, polarisation_by_sky, sky_stat_max)

"""Shared sky scan with an ungrouped compatibility entry point."""

import numpy as np
from numba import njit
from .sky_scan_scratch import scan_sky_scratch
from .sky_scan_delay import make_delay_groups, delay_groups_for_grid


@njit(cache=True)
def scan_sky_for_best_fit(
    n_ifo,
    n_pix,
    n_sky,
    FP,
    FX,
    rms,
    td00,
    td90,
    ml,
    REG,
    netCC,
    delta_regulator,
    network_energy_threshold,
    sky_valid_indices,
):
    """Compatibility scan: one group per direction, original argument contract."""
    order = np.arange(n_sky, dtype=np.int64)
    offsets = np.arange(n_sky + 1, dtype=np.int64)
    return scan_sky_scratch(
        n_ifo,
        n_pix,
        n_sky,
        FP,
        FX,
        rms,
        td00,
        td90,
        ml,
        REG,
        netCC,
        delta_regulator,
        network_energy_threshold,
        sky_valid_indices,
        order,
        offsets,
    )


find_optimal_sky_localization = scan_sky_for_best_fit


def scan_sky(geometry, cluster, settings, *, reuse_delays=True, setup=None, big_cluster=False):
    """Scan prepared geometry and cluster data using one compiled kernel.

    geometry is (FP, FX, ml); cluster is (rms, td00, td90); settings is
    (REG, netCC, delta_regulator, network_energy_threshold, sky_valid_indices).
    Arrays retain the shapes/dtypes of scan_sky_scratch. When supplied, setup
    owns the immutable geometry cache. False reuse_delays uses singleton groups.
    """
    FP, FX, ml = geometry
    rms, td00, td90 = cluster
    groups = (
        make_delay_groups(ml, reuse_delays)
        if setup is None
        else delay_groups_for_grid(setup, ml, big_cluster, reuse_delays)
    )
    return scan_sky_scratch(ml.shape[0], rms.shape[0], ml.shape[1], FP, FX, rms, td00, td90, ml, *settings, *groups)

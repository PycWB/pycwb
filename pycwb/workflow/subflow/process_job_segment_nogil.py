"""Experimental GIL-releasing lag threads, separate from process execution.

These wrappers preserve the native arithmetic. Only selected compiled entry
points release the GIL; the complete Python lag loop is not GIL-free.
"""

import importlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pycwb.utils.function_binding import specialize as _specialize

from numba import njit

from pycwb.modules.coherence_native import clustering, selection

from pycwb.modules.coherence_native.kernels import (
    _align_threshold_map_numba,
    _align_threshold_map_preindexed_numba,
    _label_components_grid,
    _select_candidates_numba,
    _subnet_subrho_batch_numba,
)
from pycwb.modules.coherence_native.run_clustering import label_components_runs
from pycwb.workflow.subflow import process_job_segment_native as native

from pycwb.modules.likelihoodWP.sky_scan import scan_sky, scan_sky_kernel

pipeline = importlib.import_module("pycwb.modules.coherence_native.coherence")


# Wrappers call the original compiled arithmetic without changing its flags.
# Only the Python-to-Numba boundary releases the GIL. Unlike replacing module
# globals, the specialized call graph below cannot affect native concurrent work.
@njit(cache=True, nogil=True)
def _align_nogil(*args):
    return _align_threshold_map_numba(*args)


@njit(cache=True, nogil=True)
def _align_preindexed_nogil(*args):
    return _align_threshold_map_preindexed_numba(*args)


@njit(cache=True, nogil=True)
def _select_nogil(*args):
    return _select_candidates_numba(*args)


@njit(cache=True, nogil=True)
def _label_grid_nogil(*args):
    return _label_components_grid(*args)


@njit(cache=True, nogil=True)
def _label_runs_nogil(*args):
    return label_components_runs(*args)


@njit(cache=True, nogil=True)
def _subnet_nogil(*args):
    return _subnet_subrho_batch_numba(*args)


_select_pixels = _specialize(
    selection.select_network_pixels,
    _align_threshold_map_numba=_align_nogil,
    _align_threshold_map_preindexed_numba=_align_preindexed_nogil,
    _select_candidates_numba=_select_nogil,
)
_cluster_pixels = _specialize(
    clustering.cluster_pixels,
    _label_components_grid=_label_grid_nogil,
    label_components_runs=_label_runs_nogil,
    _subnet_subrho_batch_numba=_subnet_nogil,
)
_coherence = _specialize(
    pipeline.coherence_single_lag,
    select_network_pixels=_select_pixels,
    cluster_pixels=_cluster_pixels,
)
_analyze_nogil = _specialize(native._run_lag_analysis, coherence_single_lag=_coherence)

_subnet_module = importlib.import_module(
    "pycwb.modules.super_cluster_native.sub_net_cut"
)
_super_module = importlib.import_module(
    "pycwb.modules.super_cluster_native.super_cluster"
)
_super_utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
_likelihood_module = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
_subnet_sky = _subnet_module.optimize_sky_loc_from_td
_subnet_mra = _subnet_module.mra_statistics_from_td


@njit(cache=True, nogil=True)
def _subnet_sky_nogil(*args):
    return _subnet_sky(*args)


@njit(cache=True, nogil=True)
def _subnet_mra_nogil(*args):
    return _subnet_mra(*args)


@njit(cache=True, nogil=True)
def _sky_scan_nogil(*args):
    return scan_sky_kernel(*args)


_subnet_packets = _specialize(
    _subnet_module._sub_net_cut_prepared_packets,
    optimize_sky_loc_from_td=_subnet_sky_nogil,
    mra_statistics_from_td=_subnet_mra_nogil,
)
_subnet_arrays = _specialize(
    _subnet_module.sub_net_cut_from_pixel_arrays,
    _sub_net_cut_prepared_packets=_subnet_packets,
)
_apply_subnet = _specialize(
    _super_utils.apply_subnet_cut, sub_net_cut_from_pixel_arrays=_subnet_arrays
)
_supercluster = _specialize(
    _super_module.supercluster_single_lag, apply_subnet_cut=_apply_subnet
)
_likelihood = _specialize(
    _likelihood_module.evaluate_cluster_likelihood,
    _scan_sky=_specialize(scan_sky, scan_sky_kernel=_sky_scan_nogil),
)
_analyze_nogil_full = _specialize(
    native._run_lag_analysis,
    coherence_single_lag=_coherence,
    supercluster_single_lag=_supercluster,
    evaluate_cluster_likelihood=_likelihood,
)


def process_lags(context, output_context, pending_lags, workers, *, full=False):
    """Run the selected private call graph with shared Python input objects."""
    from pycwb.workflow.subflow.process_job_segment_parallel import _consume_bounded

    analyze = _analyze_nogil_full if full else _analyze_nogil
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="lag-nogil"
    ) as executor:
        _consume_bounded(
            executor, partial(analyze, context), pending_lags, output_context, workers
        )

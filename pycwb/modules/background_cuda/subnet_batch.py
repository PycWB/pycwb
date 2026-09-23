"""Batch independent subnet sky scans; retain native CPU MRA/cut semantics.

Only the per-cluster sky scan (``optimze_sky_loc_from_td``) moves to the GPU,
as one :meth:`subnet_scan.SubnetScan.scan_many` call for all clusters of a
lag. Pixel loading, cross-talk, MRA and the subnet/subrho/subthr decisions run
the unchanged native code.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

import numpy as np

from . import flags
from .binding import specialize
from .subnet_scan import SubnetScan

native = importlib.import_module("pycwb.modules.super_cluster_native.sub_net_cut")
utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
logger = logging.getLogger(__name__)


class BatchedSubnet:
    """Drop-in for ``super_cluster_native.utils.apply_subnet_cut`` with a batched sky scan.

    Bound by ``processor.py`` as ``apply_subnet_cut`` inside a specialized
    ``supercluster_single_lag`` when ``PYCWB_GPU_SUBNET_BATCH=1``.

    Attributes
    ----------
    scan : SubnetScan
        Owns the resident geometry cache for the trial.
    """

    def __init__(self) -> None:
        self.scan = SubnetScan()

    def __call__(
        self,
        superclusters: list[Any],
        n_loudest_local: int,
        ml_local: np.ndarray,
        FP_local: np.ndarray,
        FX_local: np.ndarray,
        acor_local: float,
        e2or_local: float,
        n_ifo_local: int,
        n_sky_local: int,
        subnet_local: float,
        subcut_local: float,
        subnorm_local: float,
        subrho_local: float,
        xtalk_local: Any,
        arrays_prepared: bool = False,
    ) -> list[Any]:
        """Apply the subnet cut to every supercluster of one lag.

        Parameters
        ----------
        superclusters : list
            Clusters whose ``cluster_status`` is set in place (``-1`` passed,
            ``1`` rejected).
        n_loudest_local : int
            Number of loudest pixels scanned per cluster.
        ml_local : numpy.ndarray
            Sky-delay indices, shape ``(n_ifo, n_sky)``; used as int32.
        FP_local, FX_local : numpy.ndarray
            Antenna patterns. Shape ``(n_sky, n_ifo)`` when ``arrays_prepared``,
            otherwise ``(n_ifo, n_sky)`` and transposed here; used as float32.
        acor_local, e2or_local : float
            Native thresholds; ``2 * acor**2 * n_ifo`` (float32) is the pixel
            energy threshold handed to the scan.
        n_ifo_local, n_sky_local : int
            Detector count and sky-grid size.
        subnet_local, subcut_local, subnorm_local, subrho_local : float
            Native cut parameters, passed through unchanged.
        xtalk_local
            Cross-talk catalog passed through to the native packet code.
        arrays_prepared : bool, optional
            Whether ``FP_local``/``FX_local`` are already ``(n_sky, n_ifo)``.

        Returns
        -------
        list
            The surviving superclusters (``cluster_status <= 0``), in order.

        Raises
        ------
        AssertionError
            Under ``PYCWB_GPU_VALIDATE_STAGES=1``, if a GPU sky result differs
            from the native scan before the packet cuts.
        RuntimeError
            If the native packet code consumed fewer scan results than were
            produced, which would indicate a control-flow change.
        """
        start = time.perf_counter()
        FP = np.ascontiguousarray(FP_local if arrays_prepared else FP_local.T, dtype=np.float32)
        FX = np.ascontiguousarray(FX_local if arrays_prepared else FX_local.T, dtype=np.float32)
        ml = np.ascontiguousarray(ml_local, dtype=np.int32)
        rows = [utils._top_loudest_indices(c.pixel_arrays.likelihood, n_loudest_local) for c in superclusters]
        inputs = [
            native._load_selected_pixel_arrays(c.pixel_arrays, r) if len(r) else None
            for c, r in zip(superclusters, rows, strict=True)
        ]
        prep_s = time.perf_counter() - start
        threshold = np.float32(2 * acor_local * acor_local * n_ifo_local)
        t = time.perf_counter()
        results = iter(
            self.scan.scan_many(
                [x for x in inputs if x is not None],
                n_ifo_local,
                n_sky_local,
                FP,
                FX,
                ml,
                threshold,
                e2or_local,
                subcut_local,
            )
        )
        sky_s = time.perf_counter() - t

        # The original packet postprocessing invokes its scan exactly once, so
        # handing out the precomputed results in order is safe.
        def ready(*args: Any) -> tuple[Any, ...]:
            actual = next(results)
            if flags.enabled("VALIDATE_STAGES"):
                from .validation import leaves

                expected = native.optimze_sky_loc_from_td(*args)
                if leaves(expected) != leaves(actual):
                    raise AssertionError("Subnet sky result differs before packet cuts")
                logger.info("GPU subnet sky parity: exact=1")
            return actual

        packets = specialize(native._sub_net_cut_prepared_packets, optimze_sky_loc_from_td=ready)
        timing: dict[str, float] = {}
        for cluster, r, data in zip(superclusters, rows, inputs, strict=True):
            if data is None:
                # Native empty-input handling does not invoke the scan.
                result = native.sub_net_cut_from_pixel_arrays(
                    cluster.pixel_arrays,
                    r,
                    ml,
                    FP,
                    FX,
                    acor_local,
                    e2or_local,
                    n_ifo_local,
                    n_sky_local,
                    subnet_local,
                    subcut_local,
                    subnorm_local,
                    subrho_local,
                    xtalk_local,
                    arrays_prepared=True,
                    timing=timing,
                )
            else:
                result = packets(
                    *data,
                    ml,
                    FP,
                    FX,
                    acor_local,
                    e2or_local,
                    n_ifo_local,
                    n_sky_local,
                    subnet_local,
                    subcut_local,
                    subnorm_local,
                    subrho_local,
                    xtalk=xtalk_local,
                    layers=cluster.pixel_arrays.layers[r],
                    times=cluster.pixel_arrays.time[r],
                    timing=timing,
                )
            cluster.cluster_status = (
                -1 if result["subnet_passed"] and result["subrho_passed"] and result["subthr_passed"] else 1
            )
            if flags.enabled("VALIDATE_STAGES"):
                logger.info(
                    "GPU subnet decisions: subnet=%d subrho=%d subthr=%d",
                    result["subnet_passed"],
                    result["subrho_passed"],
                    result["subthr_passed"],
                )
        # Catch unexpected control-flow changes rather than shifting cluster results.
        if next(results, None) is not None:
            raise RuntimeError("Unconsumed subnet results")
        logger.info(
            "GPU batched subnet: clusters=%d prep=%.4fs sky=%.4fs xtalk=%.4fs mra=%.4fs total=%.4fs",
            len(superclusters),
            prep_s,
            sky_s,
            timing.get("xtalk", 0.0),
            timing.get("mra", 0.0),
            time.perf_counter() - start,
        )
        return [c for c in superclusters if c.cluster_status <= 0]

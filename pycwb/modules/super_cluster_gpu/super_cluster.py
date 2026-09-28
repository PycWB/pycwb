"""Compose native superclustering with job-owned CUDA subnet and TD stages."""

from pycwb.types.stages import SuperclusterStage
import importlib
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster
from functools import partial
from .td_setup_parallel import build_td_inputs_cache

native = importlib.import_module("pycwb.modules.super_cluster_native.super_cluster")
__all__ = ["build_supercluster", "setup_supercluster", "build_td_inputs_cache"]


def build_supercluster(config: object | None = None) -> SuperclusterStage:
    """Return a single-lag callable owning this process's device resources."""
    options = gpu_options(config)
    bindings = {}
    if options.subnet or options.subnet_batch:
        supercluster = importlib.import_module("pycwb.modules.super_cluster_native.super_cluster")
        if options.subnet_batch:
            from pycwb.modules.super_cluster_gpu.subnet_batch import BatchedSubnet

            apply = BatchedSubnet(options)
        else:
            from pycwb.modules.super_cluster_gpu.subnet_scan import SubnetScan

            subnet = importlib.import_module("pycwb.modules.super_cluster_native.sub_net_cut")
            utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
            packets = partial(
                subnet._sub_net_cut_prepared_packets,
                sky_optimizer=SubnetScan(),
            )
            cut = partial(
                subnet.sub_net_cut_from_pixel_arrays,
                packet_cut=packets,
            )
            apply = partial(utils.apply_subnet_cut, pixel_cut=cut)
        bindings["supercluster_single_lag"] = partial(
            supercluster.supercluster_single_lag, subnet_cut=apply
        )
    if options.td:
        from pycwb.modules.super_cluster_gpu.td_vectors import GPUTimeDelays

        bindings["supercluster_single_lag"] = partial(
            bindings.get("supercluster_single_lag", native.supercluster_single_lag),
            populate_td=GPUTimeDelays(options),
        )
    return bindings.get("supercluster_single_lag", native.supercluster_single_lag)

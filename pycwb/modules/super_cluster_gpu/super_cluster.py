"""Compose native superclustering with job-owned CUDA subnet and TD stages."""
import importlib
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster
from pycwb.utils.function_binding import specialize
from .td_setup_parallel import build_td_inputs_cache

native = importlib.import_module("pycwb.modules.super_cluster_native.super_cluster")
__all__ = ["build_supercluster", "setup_supercluster", "build_td_inputs_cache"]


def build_supercluster(config=None):
    """Return a single-lag callable owning this process's device resources."""
    options = gpu_options(config)
    bindings = {}
    if options.subnet or options.subnet_batch:
        supercluster = importlib.import_module(
            "pycwb.modules.super_cluster_native.super_cluster"
        )
        if options.subnet_batch:
            from pycwb.modules.super_cluster_gpu.subnet_batch import BatchedSubnet

            apply = BatchedSubnet(options)
        else:
            from pycwb.modules.super_cluster_gpu.subnet_scan import SubnetScan

            subnet = importlib.import_module(
                "pycwb.modules.super_cluster_native.sub_net_cut"
            )
            utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
            packets = specialize(
                subnet._sub_net_cut_prepared_packets,
                optimze_sky_loc_from_td=SubnetScan(),
            )
            cut = specialize(
                subnet.sub_net_cut_from_pixel_arrays,
                _sub_net_cut_prepared_packets=packets,
            )
            apply = specialize(
                utils.apply_subnet_cut, sub_net_cut_from_pixel_arrays=cut
            )
        bindings["supercluster_single_lag"] = specialize(
            supercluster.supercluster_single_lag, apply_subnet_cut=apply
        )
    if options.td:
        from pycwb.modules.super_cluster_gpu.td_vectors import GPUTimeDelays

        bindings["supercluster_single_lag"] = specialize(
            bindings.get("supercluster_single_lag", native.supercluster_single_lag),
            _populate_td_vectors=GPUTimeDelays(options),
        )
    return bindings.get("supercluster_single_lag", native.supercluster_single_lag)

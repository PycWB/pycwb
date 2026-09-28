"""Compose the release likelihood API with selected CUDA implementations."""
import importlib
from pycwb.constants.execution_profile import execution_profile
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.likelihoodWP.likelihood import prepare_likelihood_inputs
from pycwb.utils.function_binding import specialize

# Import the module explicitly: package exports may share its name.
native = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
__all__ = ["build_likelihood", "prepare_likelihood_inputs"]


def build_likelihood(config=None):
    """Build one process's DPF, sky-scan and chirp resources using job options."""
    options = gpu_options(config)
    bindings = {}
    if options.dpf:
        if not execution_profile(config).scalar_dpf:
            raise ValueError("gpu.dpf requires execution_profile.scalar_dpf=true")
        from pycwb.modules.likelihood_gpu.dpf_regulator import DPFRegulator

        likelihood_module = importlib.import_module(
            "pycwb.modules.likelihoodWP.likelihood"
        )
        bindings["evaluate_cluster_likelihood"] = specialize(
            likelihood_module.evaluate_cluster_likelihood,
            _compute_dpf_regulator_scalar=DPFRegulator(options),
        )
    if options.likelihood:
        from pycwb.modules.likelihood_gpu.likelihood_scan import LikelihoodScan

        scan = LikelihoodScan(options)
        bindings["evaluate_cluster_likelihood"] = specialize(
            bindings.get(
                "evaluate_cluster_likelihood", native.evaluate_cluster_likelihood
            ),
            _scan_sky=scan.scan_sky,
        )
    if options.chirp:
        from pycwb.modules.likelihood_gpu.chirp_bootstrap import make_chirp_update

        bindings["evaluate_cluster_likelihood"] = specialize(
            bindings.get(
                "evaluate_cluster_likelihood", native.evaluate_cluster_likelihood
            ),
            _update_cluster_chirp_statistics=make_chirp_update(),
        )
    return bindings.get("evaluate_cluster_likelihood", native.evaluate_cluster_likelihood)

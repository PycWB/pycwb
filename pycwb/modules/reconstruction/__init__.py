from importlib import import_module


_EXPORTS = {
    "get_network_MRA_wave": (
        "pycwb.modules.reconstruction.getMRAwaveform",
        "get_network_MRA_wave",
    ),
    "get_INJ_waveform": (
        "pycwb.modules.reconstruction.getINJwaveform",
        "get_INJ_waveform",
    ),
    "estimate_snr": ("pycwb.modules.reconstruction.getINJwaveform", "estimate_snr"),
    "get_residuals": ("pycwb.modules.reconstruction.getResiduals", "get_residuals"),
    "get_ASD": ("pycwb.modules.reconstruction.getResiduals", "get_ASD"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

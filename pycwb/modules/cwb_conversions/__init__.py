from importlib import import_module

_EXPORTS = {
    "convert_to_wseries": "series",
    "convert_to_wavearray": "series",
    "convert_numpy_to_wavearray": "wavearray",
    "convert_timeseries_to_wavearray": "series",
    "convert_pycwb_timeseries_to_wavearray": "series",
    "convert_pycbc_timeseries_to_wavearray": "series",
    "WSeries_to_matrix": "series",
    "convert_wavearray_to_timeseries": "series",
    "convert_wavearray_to_pycwb_timeseries": "series",
    "convert_wavearray_to_pycbc_timeseries": "series",
    "convert_wavearray_to_nparray": "series",
    "convert_wseries_to_timeseries": "series",
    "convert_wseries_to_pycwb_timeseries": "series",
    "convert_wseries_to_pycbc_timeseries": "series",
    "convert_wseries_to_time_frequency_series": "series",
    "convert_time_frequency_series_to_wseries": "series",
    "convert_fragment_clusters_to_netcluster": "cluster",
    "convert_cluster_meta_to_cData": "cluster",
    "convert_netcluster_to_fragment_clusters": "cluster",
    "convert_netcluster_to_cluster": "cluster",
    "convert_cData_to_cluster_meta": "cluster",
    "convert_pixel_to_netpixel": "pixel",
    "convert_to_pixdata": "pixel",
    "convert_netpixel_to_pixel": "pixel",
    "convert_td_amp_to_cwb": "pixel",
    "convert_sparse_series_to_sseries": "sparse_series",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

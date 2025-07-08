import numpy as np
import ROOT
import cppyy


def convert_sparse_series_to_sseries(sparse_series):
    ss = ROOT.SSeries(np.double)()
    ss.pWavelet = sparse_series.wavelet.lightweight_dump().wavelet
    data = np.array(sparse_series.sparse_map_00 + sparse_series.sparse_map_90, dtype=np.double)
    if len(data) == 0:
        ss.pWavelet.allocate(0, ROOT.nullptr)
    else:
        _ptr = cppyy.gbl._to_double_malloc(data, len(data))
        ss.pWavelet.allocate(len(data), _ptr)

    ss.sparseLookup.resize(len(sparse_series.sparse_lookup))
    for i, v in enumerate(sparse_series.sparse_lookup):
        ss.sparseLookup[i] = v
    ss.sparseIndex.resize(len(sparse_series.sparse_index))
    for i, v in enumerate(sparse_series.sparse_index):
        ss.sparseIndex[i] = v
    ss.sparseMap00.resize(len(sparse_series.sparse_map_00))
    for i, v in enumerate(sparse_series.sparse_map_00):
        ss.sparseMap00[i] = v
    ss.sparseMap90.resize(len(sparse_series.sparse_map_90))
    for i, v in enumerate(sparse_series.sparse_map_90):
        ss.sparseMap90[i] = v
    ss.rate(sparse_series.rate)
    ss.wrate(sparse_series.w_rate)
    ss.start(sparse_series.start)
    ss.stop(sparse_series.stop)
    ss.edge(sparse_series.edge)
    ss.time_Halo = sparse_series.time_halo
    ss.layerHalo = sparse_series.layer_halo
    ss.net_Delay = sparse_series.net_delay
    return ss

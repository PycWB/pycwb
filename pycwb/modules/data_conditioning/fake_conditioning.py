from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_wseries_to_time_frequency_series
from pycwb.types.wdm import WDM
from pycwb.types.time_series import TimeSeries
import numpy as np
import h5py
try:
    import ROOT
except ImportError:
    ROOT = None
    import warnings
    warnings.warn(
        "ROOT module not found. CWB conversions will not work. This warning will be removed in future versions when ROOT is no longer a dependency.",
        ImportWarning,
        stacklevel=2
    )

def fake_conditioning(config, strains):
    """
    Performs fake data conditioning for bootstrap analysis by generating fake nRMS and applying it to the input strains

    :param config: config object
    :type config: Config
    :param strains: list of strain data
    :type strains: list[TimeSeries | gwpy.timeseries.TimeSeries | ROOT.wavearray(np.double)]
    :return: (conditioned_strains, nRMS_list)
    :rtype: tuple[list[TimeFrequencySeries], list[TimeFrequencySeries]]
    """
    nRMS_list = []
    conditioned_strains = []
    
    for i, ifo in enumerate(config.ifo): 
        #Generate fake nRMS for the current IFO
        layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
        wdm = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision)
        
        #load nrms and stores thes 
        nRMS = _load_timeseries_h5(config.bootstrap['nrms'][ifo])
        nRMS = generate_nrms_wseries(config, strains[i], nRMS, wdm.wavelet)
        nRMS_list.append(convert_wseries_to_time_frequency_series(nRMS))

        #Generate the time frequency series 
        tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(strains[i]), wdm.wavelet)
        tf_map.w_mode = 1   #Change w_mode to 1 for compatibility with "Network.cc" 

        conditioned_strains.append(convert_wseries_to_time_frequency_series(tf_map))
    
    return conditioned_strains, nRMS_list


def _load_timeseries_h5(path):
    """Read an HDF5 time series file.

    Returns a pycwb TimeSeries (only the data array is used downstream,
    so start_time/delta_t are best-effort).
    """
    with h5py.File(path, 'r') as f:
        # HDF5 format stores data at the root 'data' key
        if 'data' in f:
            data = np.array(f['data'], dtype=np.float64)
            t0 = float(f['data'].attrs.get('start_time', 0.0))
            dt = float(f['data'].attrs.get('delta_t', 1.0))
        elif 'strain/Strain' in f:
            data = np.array(f['strain/Strain'], dtype=np.float64)
            t0 = float(f['strain/Strain'].attrs.get('Xstart', 0.0))
            dt = float(f['strain/Strain'].attrs.get('Xspacing', 1.0))
        else:
            # Fallback: read the first dataset found
            key = list(f.keys())[0]
            data = np.array(f[key], dtype=np.float64)
            t0 = 0.0
            dt = 1.0
    return TimeSeries(data=data, t0=t0, dt=dt)


def generate_nrms_wseries(config, data, nrms, wavelet): 
    """ 
    Generates a WSeries object for the nRMS from the numpy array of nRMS values
    """ 
    #Generate the WSeries and use it to initialise nRMS with the correct parameters
    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(data), wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)

    #Generate nRMS series 
    nRMS = tf_map.white(config.whiteWindow, 0, config.segEdge, config.whiteStride)
    nRMS.bandpass(16., 0., 1)
    
    #Substitute the cWB nRMS with the MESA nRMS 
    for i in range(len(nRMS)): 
        nRMS.data[i] = nrms[i]

    return nRMS 
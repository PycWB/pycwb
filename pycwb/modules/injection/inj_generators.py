from pycwb.types.time_series import TimeSeries
from scipy.signal import resample_poly
import numpy as np
import h5py
import logging
from numpy import sqrt

logger = logging.getLogger(__name__)


def _load_timeseries(path):
    """Load a time series from HDF5, NumPy, or text file.

    Supports HDF5 files, .npy files, and whitespace-delimited
    text files.
    """
    if path.endswith('.npy'):
        data = np.load(path).astype(np.float64)
        return TimeSeries(data=data, t0=0.0, dt=1.0)
    elif path.endswith('.hdf') or path.endswith('.hdf5') or path.endswith('.h5'):
        with h5py.File(path, 'r') as f:
            if 'data' in f:
                data = np.array(f['data'], dtype=np.float64)
                t0 = float(f['data'].attrs.get('start_time', 0.0))
                dt = float(f['data'].attrs.get('delta_t', 1.0))
            elif 'strain/Strain' in f:
                data = np.array(f['strain/Strain'], dtype=np.float64)
                t0 = float(f['strain/Strain'].attrs.get('Xstart', 0.0))
                dt = float(f['strain/Strain'].attrs.get('Xspacing', 1.0))
            else:
                key = list(f.keys())[0]
                data = np.array(f[key], dtype=np.float64)
                t0 = 0.0
                dt = 1.0
        return TimeSeries(data=data, t0=t0, dt=dt)
    else:
        # text file: assume single column of data values
        data = np.loadtxt(path, dtype=np.float64)
        return TimeSeries(data=data, t0=0.0, dt=1.0)


def get_strain_from_file(delta_t, files, allow_resampling = False, **kwargs): 
    """
    Generates the strain reading it from a file. The available extensions are: .txt, .npy and .hdf.
    Parameters:
    -----------
    parameters: dict
        A dictionary containing the following keys:
        - 'sample_rate': The sample rate of the data in which the strain is to be injected. Default is 16384 Hz. 
        - 'files': A dictionary with keys as interferometer names and values as file paths. 
                    Eg {'H1': 'path/to/H1_strain.txt', 'L1': 'path/to/L1_strain.txt'}
    Returns:
    --------
    injections: dict
        A dictionary with keys as interferometer names and values as pycwb TimeSeries.
    """
    #Initialize the injections dictionary
    injections = {'type': 'strain'}
    sample_rate = 1 / delta_t 
    central_time = None 
    distribute = kwargs.get('distribute', True)
    for ifo, file in files.items():
        logger.info(f"Loading strain data for {ifo} from {file}") 
        strain = _load_timeseries(file)
        #Only compute central time once so that detector dT is preserved 
        if distribute: 
            if central_time is None: 
                central_time = compute_central_time(strain)
            
            strain.start_time = kwargs['gps_time'] - central_time 

        if kwargs.get('rescale', None):
            rescale_factor = sqrt(2) ** kwargs['rescale'] 
            logger.info(f"Rescaling strain data by a factor of sqrt(2) ** {kwargs['rescale']}")
            strain.data *= rescale_factor 

        if strain.sample_rate == sample_rate:
            injections[ifo] = strain

        #resample the data if injection sample rate and target time series sample rate do not match 
        if strain.sample_rate != sample_rate: 
            if not allow_resampling and distribute:
                raise ValueError(f"Strain sample rate ({strain.sample_rate} Hz) does not match target sample rate ({sample_rate} Hz). Set allow_resampling = True to enable resampling, or set distribute = False to disable resampling and preserve original sample rate.")
            else: 
                logger.warning(f"Resampling {ifo} data with a polyphase filter from {strain.sample_rate} to {sample_rate}")
                factor = sample_rate / strain.sample_rate 
                strain = resample_data(strain, factor)

        injections[ifo] = strain  
    return injections

def resample_data(data, factor): 
    """
    Resamples the data using a polyphase filter.
    
    Parameters:
    -----------
    data: np.ndarray
        The input data to be resampled.
    factor: int
        The resampling factor.
        
    Returns:
    --------
    np.ndarray
        The resampled data.
    """
    if factor > 1:   # Upsample
        resampled_data = resample_poly(data.data, int(factor), 1)

    elif factor < 1: # Downsample 
        resampled_data = resample_poly(data.data, 1, int(1 / factor))

    else:            # No resampling needed
        resampled_data = data.data 
    resampled_data = TimeSeries(data=resampled_data, t0=float(data.start_time), dt=data.delta_t / factor)
    return resampled_data


def compute_central_time(strain): 
    """ 
    Computes the central time of the strain data.
    """ 
    return (strain.data * strain.data * strain.sample_times.data).sum() / (strain.data * strain.data).sum() - strain.sample_times.data[0] 

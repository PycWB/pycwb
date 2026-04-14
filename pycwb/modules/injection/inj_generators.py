"""Injection generators for reading strain data from files."""

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


def _load_wave_timeseries(file, id, ifo, label):
    """Load a time series from a wave file.

    Parameters
    ----------
    files : dict
        Mapping of interferometer names to file paths, e.g.
        ``{'H1': 'path/to/catalog.parquet'}``.
    id : str
        The unique identifier for the event to load from the catalog.
    ifo : str
        The interferometer name (e.g., 'H1', 'L1', 'V1').
    label : str
        The type of waveform to be loaded available ('REC', 'INJ', 'DAT', 'NUL')
    """ 
    logger.info(f'Creating time series from wave file: {file}, id: {id}, channel: {ifo}_wf_{label}')

    with h5py.File(file, 'r') as f: 
        wave = f[id] 
        strain = wave[f'{ifo}_wf_{label}'][:]
        sample_rate = wave[f'{ifo}_wf_{label}'].attrs['sample_rate']
        start_time = wave[f'{ifo}_wf_{label}'].attrs['start_time']

    return TimeSeries(strain, dt =1/sample_rate, t0=start_time) 


def get_strain_from_file(delta_t, files, allow_resampling = False, **kwargs): 
    """Generate strain by reading it from a file.

    The available extensions are: .txt, .npy and .hdf.

    Parameters
    ----------
    delta_t : float
        Sampling interval (1 / sample_rate).
    files : dict
        Mapping of interferometer names to file paths, e.g.
        ``{'H1': 'path/to/H1_strain.txt', 'L1': 'path/to/L1_strain.txt'}``.
    allow_resampling : bool, optional
        If ``True``, resample data whose sample rate does not match the
        target rate.  Default is ``False``.
    **kwargs
        Additional injection parameters (``gps_time``, ``rescale``,
        ``distribute``, etc.).

    Returns
    -------
    dict
        A dictionary with keys as interferometer names and values as
        pycwb TimeSeries.
    """
    #Initialize the injections dictionary
    injections = {'type': 'strain'}
    sample_rate = 1 / delta_t 
    central_time = None 
    distribute = kwargs.get('distribute', True) 

    if kwargs.get('is_wave_file', False):  
       if kwargs.get('id', None) is None or kwargs.get('label', None) is None:
           logger.error(f"When 'is_wave_file' is True, both 'id' and 'label' must be provided, given values are id: {kwargs.get('id', None)}, label: {kwargs.get('label', None)}")
           raise ValueError(f"When 'is_wave_file' is True, both 'id' and 'label' must be provided, given values are id: {kwargs.get('id', None)}, label: {kwargs.get('label', None)}") 
       
       strains = {ifo: _load_wave_timeseries(file, id=kwargs['id'], ifo=ifo, label=kwargs['label']) for ifo, file in files.items()} 
    
    else: 
        strains = {ifo: _load_timeseries(file) for ifo, file in files.items()}
    
    for ifo, strain in strains.items():
        #Only compute central time once so that detector dT is preserved 
        if distribute: 
            if central_time is None: 
                central_time = compute_central_time(strain)
            
            strain.start_time = kwargs['gps_time'] - central_time 

        if kwargs.get('rescale', None):
            rescale_factor = sqrt(2) *1.2005065821456904e-21* kwargs['rescale'] 
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
    """Resample data using a polyphase filter.

    Parameters
    ----------
    data : TimeSeries
        The input time series to be resampled.
    factor : float
        The resampling factor (>1 upsamples, <1 downsamples).

    Returns
    -------
    TimeSeries
        The resampled time series.
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
    """Compute the central time of the strain data."""
    return (strain.data * strain.data * strain.sample_times.data).sum() / (strain.data * strain.data).sum() - strain.sample_times.data[0] 

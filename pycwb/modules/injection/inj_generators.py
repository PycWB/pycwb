from pycbc.types.timeseries import TimeSeries, load_timeseries
from scipy.signal import resample_poly
import logging 

logger = logging.getLogger(__name__)


def get_strain_from_file(delta_t, files, allow_resampling = False, **kwargs): 
    """
    Generates the pycbc strain reading it from a file. The available extensions are: .txt, .npy and .hdf.
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
        A dictionary with keys as interferometer names and values as pycbc TimeSeries.
    """
    #Initialize the injections dictionary
    injections = {'type': 'strain'}
    sample_rate = 1 / delta_t 

    for ifo, file in files.items():
        logger.info(f"Loading strain data for {ifo} from {file}") 
        strain = load_timeseries(file)
        strain.start_time = kwargs['gps_time']

    
        if strain.sample_rate == sample_rate:
            injections[ifo] = strain

        elif not allow_resampling and strain.sample_rate != sample_rate:
            raise ValueError(f"Strain sample rate ({strain.sample_rate} Hz) does not match target sample rate ({sample_rate} Hz). Set allow_resampling = True to enable resampling.")

        else: 
            logger.warning(f"Resampling {ifo} data with a polyphase filter from {strain.sample_rate} to {sample_rate}")
            factor = sample_rate / strain.sample_rate
            injections[ifo] = resample_data(strain, factor)

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
    resampled_data = TimeSeries(resampled_data, delta_t= data.delta_t / factor, epoch=data.start_time)
    return resampled_data

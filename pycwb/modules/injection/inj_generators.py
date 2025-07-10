from pycbc.types.timeseries import TimeSeries, load_timeseries
from scipy.signal import resample_poly
import logging 

logger = logging.getLogger(__name__)


def get_strain_from_file(**parameters): 
    """
    Generates the pycbc strain reading it from a file. The available extensions are: .txt, .npy and .hdf5.
    Parameters:
    -----------
    parameters: dict
        A dictionary containing the following keys:
        - 'files': A dictionary with keys as interferometer names (e.g., 'H1', 'L1') and values as file paths.
        - 'start_time': The GPS start time for the strain data.
    Returns:
    --------
    injections: dict
        A dictionary with keys as interferometer names and values as pycbc strain objects.
    """
    #Initialize the injections dictionary
    injections = {'type': 'strain'}
    parameters.setdefault('sample_rate', 16384)  # Default target sample rate


    #Add options for different extensions? npy, txt 
    for ifo, file in parameters['files'].items():
        logger.info(f"Loading strain data for {ifo} from {file}") 
        strain = load_timeseries(file)
        strain.start_time = parameters['gps_time']
        #Re-Sample signal if actual rate is different from the target sample rate in parameters 
        factor = parameters['sample_rate'] / strain.sample_rate
        if factor != 1: 
            logger.info(f"Resampling {ifo} data with factor {factor:.2f} (from {strain.sample_rate} to {parameters['sample_rate']})")
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
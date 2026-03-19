from pycwb.types.time_series import TimeSeries 
from memspectrum import MESA 
import numpy as np 
import logging

logger = logging.getLogger(__name__)
 

def get_residuals(strain, waveform, inRate, full_segment = False, rescale = False): 
    """    Calculate the residuals between the data and the waveform.
    """ 
    strain = strain.copy() 
    waveform = waveform.copy() 
    scale_factor = 1. / np.sqrt(2) ** (np.log2(inRate / strain.sample_rate))
    waveform /= scale_factor
    # If full_segment is False, slice the strain to the same time range as the waveform before calculating residuals
    if not full_segment:
        strain_sliced = strain.time_slice(waveform.start_time, waveform.end_time) 
        strain_residuals = strain_sliced - waveform

    #If full_segment is True, calculate residuals over the full segmen
    else:  
        start_idx, end_idx = get_indeces(strain, waveform.start_time, waveform.end_time, mode='floor')  
        logger.info(f"Calculated start and end indices for slicing: start_idx={start_idx}, end_idx={end_idx}")
        #create a padded waveform to avoid ValueError arising from time-series misalignment 
        waveform_padded = TimeSeries(data=np.zeros_like(strain.data), t0=strain.t0, dt=strain.dt)
        waveform_padded.data[start_idx:end_idx] = waveform.data
        strain_residuals = strain - waveform_padded
    if rescale:
        strain_residuals *= scale_factor

    return strain_residuals



def get_ASD(residuals, method = 'Fast', m = 1500):
    """
    Calculate the one-sided amplitude spectral density (ASD) of the residuals.
    """
    M = MESA() 
    M.solve(residuals.data - np.mean(residuals.data), m = m, method = method) 
    frequency, spectrum = M.spectrum(residuals.delta_t) 
    onesided_asd = np.sqrt(spectrum[frequency > 0] * 2) 

    return np.transpose(np.array([frequency[frequency > 0], onesided_asd])) 


def get_indeces(timeseries, start, end, mode='floor'):
        """
        Return the slice of the time series that contains the time range
        in GPS seconds.
        """
        if start < timeseries.start_time:
            raise ValueError('Time series does not contain a time as early as %s' % start)

        if end > timeseries.end_time:
            raise ValueError('Time series does not contain a time as late as %s' % end)

        start_idx = float(start - timeseries.start_time) * timeseries.sample_rate
        end_idx = float(end - timeseries.start_time) * timeseries.sample_rate

        if np.isclose(start_idx, round(start_idx), rtol=0, atol=1E-3):
            start_idx = round(start_idx)

        if np.isclose(end_idx, round(end_idx), rtol=0, atol=1E-3):
            end_idx = round(end_idx)

        if mode == 'floor':
            start_idx = int(start_idx)
            end_idx = int(end_idx)
        elif mode == 'nearest':
            start_idx = int(round(start_idx))
            end_idx = int(round(end_idx))
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        return start_idx, end_idx
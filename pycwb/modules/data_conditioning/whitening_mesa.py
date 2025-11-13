import numpy as np 
import ROOT
import logging
from pycwb.modules.cwb_conversions import *
from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.types.wdm import WDM
from pycbc.types.timeseries import TimeSeries 
from functools import partial
import multiprocessing as mp
from memspectrum import MESA
import os 
from scipy import signal
from scipy.special import expit 
from pycbc.filter.resample import highpass
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


def whitening_mesa(config, h):
    """
    Performs whitening on the given strain data
    
    Parameters
    ----------
    config: config object
            The config file for the analysis 
    h:      pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries
            The timeSeries containing the Data  
    """ 
    
    #Initialise WDM
    logger.info(f'Whitening the data using Maximum Entropy Spectral Analysis.') 
    logger.info(f"autoregressive order = {config.mesaOrder}, solving method = {config.mesaSolver}") 

    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision)
    wdmFlen = wdm_white.m_H / config.rateANA
    
    if config.mesaStride != config.mesaWindow / 3: 
        logger.warning('Value of mesa Stide in config must be one third of mesa Window. It will be changed in the analysis')
        config.mesaStride = config.mesaWindow / 3 
    #Sampling variables 
    sampling_rate = config.inRate / (2 ** config.levelR)
    Ny = .5 * sampling_rate     #The Nyquist frequency 
    wdm_df = Ny / layers_white  #Frequency resolution for the WDM 
    wdm_dt = .5 / wdm_df        #Time resolution for the WDM 
    

    #Convert to pycbc and remove mean 
    try: 
        h = h.to_pycbc() if type(h) == TimeSeries else h 
    except AttributeError: 
        pass 
    h -= np.mean(h) 

    #Filter the data 
    a,b = signal.butter(8, config.fLow / Ny, btype = 'high', analog = False) 
    h.data = signal.filtfilt(a,b,h)
   
    
    #Initialise whitening
    M = MESA() 
    stride = int(config.mesaStride * sampling_rate)
    window = int(config.mesaWindow * sampling_rate)
     
    psds = [] 
    h_white = h.copy() 
    n_windows = (len(h) - window) // stride
    #Loop over data segment chunks to compute PSDs 
    for i in range(n_windows + 1):
        start = i * stride
        M.solve(h[start:start+ window], method = config.mesaSolver, m = config.mesaOrder)
        f, psd = M.spectrum(1 / sampling_rate) 
        psds.append(psd) 

    #Smooth the PSDs estimates with a rolling median filter
    if config.mesaHalfSeg > 0: 
        logger.info(f"Smoothing PSDs estimates with rolling median filter over {config.mesaHalfSeg * 2 + 1} segments")
        psds = rolling_median(np.array(psds), half_size = config.mesaHalfSeg)

    #reindex psds to exclude possibly-glitch-contaminated estimates 
    if config.mesaReindex: 
        logger.warning("Reindexing PSDs estimates with Isolation Forest to exclude possible glitch contamination. To disable set mesaReindex to False in config file")
        psds = reindex_psds(np.array(psds), f)

    W = planck_taper_window(window) 

    #Use PSDs estimates to whiten the data     
    for i in range(n_windows + 1):
        start = i  * stride 
        stop = start + window
        
        #get indeces for whitening segments [w] and segment to be whitened [s]
        h_tmp = h[start:stop] * W
        h_w = (h_tmp.to_frequencyseries() / psds[i][:len(h_tmp) // 2 + 1]**0.5).to_timeseries() * np.sqrt(1 / sampling_rate)
        
        if i == 0: 
            h_white[start:stop-stride] = h_w[:-stride]
        if i == n_windows: 
            h_white[start+stride:stop] = h_w[stride:]
        else: 
            h_white[start + stride:stop-stride] = h_w[stride:-stride]
    del(h_tmp) 
    del(h_w) 

    #Initialize TF map 
    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(h[:stop]), wdm_white.wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)
 
    #Create a TF map for whitened array 
    tf_map_white = ROOT.WSeries(np.double)(convert_to_wavearray(h_white[:stop]), wdm_white.wavelet)
    tf_map_white.Forward()
    tf_map_white.setlow(config.fLow)
    tf_map_white.sethigh(config.fHigh)

    #Compute effective whitening filter 
    nRMS_matrix = WSeries_to_matrix(tf_map) / WSeries_to_matrix(tf_map_white)
    
    #Compute the nRMS taking the median over "whiteStride" seconds segments and convert to array 
    data_per_batch = int(config.whiteStride // wdm_dt)
    nRMS_reshaped = nRMS_matrix.reshape(int(Ny / wdm_df)+1,-1,data_per_batch) 
    nRMS_reshaped[:int(16 / wdm_df) + 1] = 1     
    
    #Compute the nRMS as the median over the segments ignoring nans if present
    nRMS = np.sqrt(np.nanmedian(nRMS_reshaped ** 2, axis = 2))
    nRMS = generate_nrms_wseries(config, h, nRMS.reshape(-1, order = 'F'))
    
    #Convert to tf map types
    tf_map_whitened = ROOT.WSeries(np.double)(convert_to_wavearray(h_white), wdm_white.wavelet)
    tf_map_whitened.w_mode = 1   #Change w_mode to 1 for compatibility with "Network.cc" 
    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map_whitened)
    n_rms = convert_wseries_to_time_frequency_series(nRMS)
    
    return tf_map_whitened, n_rms


def rolling_median(psds, half_size): 
    """ 
    Smooths the PSDs estimates with a rolling median filter 

    Parameters: 
    ----------
    psds:   np.ndarray (N_segments, N_frequencies)      
            array containing all the PSDs estimates 
    half_size: int, optional
            Half size of the rolling median window. The number of segments is The default is 2 * half_size + 1. 
            Default value is 4 
    """ 

    out = np.empty_like(psds)
    N = psds.shape[0]
    
    for i in range(N):
        #Define left and right edges of the rolling median window
        left = i - half_size
        right = i + half_size + 1  

        # If we run off the left edge, extend to the right to compensate
        if left < 0:
            deficit = -left
            left = 0
            right = min(N, right + deficit)

        # If we run off the right edge, extend to the left to compensate
        if right > N:
            deficit = right - N
            right = N
            left = max(0, left - deficit)

        out[i] = np.median(psds[int(left):int(right)], axis=0)

    return out


def reindex_psds(psds, f): 
    """ 
    Uses Isolation Forest to exclude PSDs estimates with a too large deviation from the median to exclude glitch contamination 

    Parameters: 
    ----------
    psds:   np.ndarray (N_segments, N_frequencies)      
            array containing all the PSDs estimates 
    f:      np.ndarray (N_frequencies,) 
            array containing the sampling frequencies 
    """ 

    #Compute ratio of PSDs over median over some frequencies bin to have a reference PSD 
    mask_lf = (f > 16) & (f < 128)
    mask_hf = (f > 128)
    psd_median = np.median(psds, axis = 0)

    #Compute the log distance over low and high frequency ranges to find IF features  
    dist_lf = np.mean((np.log(psds[:,mask_lf] / psd_median[mask_lf])) ** 2, axis = 1) ** .5
    dist_hf = np.mean((np.log(psds[:,mask_hf] / psd_median[mask_hf])) ** 2, axis = 1) ** .5
    dist = np.stack([dist_lf, dist_hf], axis = 1)
    predictions = IsolationForest(n_estimators = 100, contamination = .15).fit_predict(dist)
    
    #Remove outliers if found below the median  
    median_lf, median_hf = np.median([dist_lf,dist_hf], axis = 1)
    above_median = (dist_lf > median_lf) | (dist_hf > median_hf)
    predictions = np.logical_and(predictions == -1,above_median)

    #Reindex outliers taking the "closest in time" - non deviating - PSD estimate 
    outliers_idx = np.where(predictions)[0] 
    inliers_idx = np.where(~predictions)[0] 
    for idx in outliers_idx: 
        new_idx = inliers_idx[np.argmin(np.abs(inliers_idx-idx))]
        psds[idx] = psds[new_idx]
    logger.info(f"Reindexed {len(outliers_idx)} PSDs estimates out of {psds.shape[0]} total segments")

    return psds

def planck_taper_window(N, eps = 0.15):
    window = np.zeros(N)
    for k in range(N):
        if k == 0 or k == N - 1:
            window[k] = 0
        elif 0 < k < eps * (N - 1):
            za = eps * (N - 1) * (1 / k + 1 / (k - eps * (N - 1)))
            window[k] = expit(-za)
        elif eps * (N - 1) <= k <= (1 - eps) * (N - 1):
            window[k] = 1
        elif (1 - eps) * (N - 1) < k < N - 1:
            zb = eps * (N - 1) * (
                1 / (N - 1 - k) + 1 / ((1 - eps) * (N - 1) - k)
            )
            window[k] = expit(-zb)
    
    return window

def generate_nrms_wseries(config, data, nrms): 
    """ 
    Generates a WSeries object for the nRMS from the numpy array of nRMS values
    """ 

    #Generates the WDM object for the nRMS
    layers = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm = WDM(layers, layers, config.WDM_beta_order, config.WDM_precision)

    #Generate the WSeries and use it to initialise nRMS with the correct parameters
    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(data), wdm.wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)
    if config.whiteWindow != 60 or config.whiteStride != 20:
        logger.warning(f"whiteWindow = {config.whiteWindow} and whiteStride = {config.whiteStride} in config are different from default values (60,20). This may lead to unexpected results in the whitening nRMS generation.")
    nRMS = tf_map.white(config.whiteWindow, 0, config.segEdge, config.whiteStride)
    nRMS.bandpass(16., 0., 1)
    
    #Substitute the cWB nRMS with the MESA nRMS 
    for i in range(len(nRMS)): 
        nRMS.data[i] = nrms[i]

    #print(len(nrms))
    #print(len(nRMS)) 

    return nRMS 

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
    logger.info(f'Whitening the data using Maximum Entropy Spectral Analysis. \nThe chosen parameters are:\
            autoregressive order = {config.mesaOrder}, solving method = {config.mesaSolver}') 

    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision)
    wdmFlen = wdm_white.m_H / config.rateANA
    
    if config.whiteStride != config.whiteWindow / 3: 
        logger.warning('Value of whiteStride in config must be one third of whiteWindow. It will be changed in the analysis')
        config.whiteStride = config.whiteWindow / 3 
    #Sampling variables 
    sampling_rate = config.inRate / (2 ** config.levelR)
    Ny = .5 * sampling_rate #The Nyquist frequency 
    wdm_df = Ny / layers_white #frequency resolution for the WDM 
    wdm_dt = .5 / wdm_df #time resolution for the WDM 
    

    #Convert to pycbc and remove mean 
    try: 
        h = h.to_pycbc() if type(h) == TimeSeries else h 
    except AttributeError: 
        pass 
    h -= np.mean(h) 

    #filter the data 
    a,b = signal.butter(8, config.fLow / Ny, btype = 'high', analog = False) 
    h.data = signal.filtfilt(a,b,h)
   
    #pycbc highpass filter gives problems 
    #h = h.highpass_fir(frequency = config.fLow, order = 50, remove_corrupted = False)
    
    #Initialise whitening
    M = MESA() 
    stride = int(config.whiteStride * sampling_rate)
    window = int(config.whiteWindow * sampling_rate)
     
    psds = [] 
    h_white = h.copy() 
    n_windows = (len(h) - window) // stride
    #Loop over data segment chunks to compute PSDs 
    for i in range(n_windows + 1):
        start = i * stride
        M.solve(h[start:start+ window], method = config.mesaSolver, m = config.mesaOrder)
        f, psd = M.spectrum(1 / sampling_rate) 
        psds.append(psd) 
    
    #reindex psds to exclude possibly-glitch-contaminated estimates 
    if config.mesaReindex: 
        psds = reindex_psds(np.array(psds), f)
    
    #Use PSDs estimates to whiten the data         
    for i in range(n_windows + 1):
        start = i  * stride 
        stop = start + window
        
        #get indeces for whitening segments [w] and segment to be whitened [s]
        h_tmp = h[start:stop] 
        h_w = (h_tmp.to_frequencyseries() / psds[i][:len(h_tmp) // 2 + 1]**0.5).to_timeseries() * np.sqrt(1 / sampling_rate)
        
        if i == 0: 
            h_white[start:stop-stride] = h_w[:-stride]
        if i == n_windows: 
            h_white[start+stride:stop] = h_w[stride:]
        else: 
            h_white[start + stride:stop-stride] = h_w[stride:-stride]
    del(h_tmp) 
    del(h_w) 
    edge = int(config.segEdge * sampling_rate)
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
    

    #Compute the nRMS taking the median over 20 second segments and convert to array 
    data_per_batch = int(config.whiteStride // wdm_dt)
    nRMS_reshaped = nRMS_matrix.reshape(int(Ny / wdm_df),-1,data_per_batch) 
    nRMS_reshaped[:int(16 / wdm_df)] = 1  #set nRMS = 1 for f < 16 
    nRMS = np.sqrt(np.median(nRMS_reshaped ** 2, axis = 2))

    #add 1 as last frequency bin for compatibility 
    n_segments = nRMS_reshaped.shape[1]
    nRMS = np.vstack((nRMS,np.ones((1,n_segments)))).reshape(-1, order = 'F')  
    
 
    #convert to pycwb types nRMS
    #nRMS = _convert_numpy_nrms_to_wseries(nRMS, h.start_time, sampling_rate, n_segments,  wdm_white.wavelet)
    tf_map_whitened = ROOT.WSeries(np.double)(convert_to_wavearray(h_white[:stop]), wdm_white.wavelet)
    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map_whitened)
    #n_rms = convert_wseries_to_time_frequency_series(nRMS)
    n_rms = tf_map_whitened.copy() 
    n_rms.data = TimeSeries(nRMS, delta_t = 1 / sampling_rate, epoch = tf_map_whitened.data.start_time)
    
    return tf_map_whitened, n_rms


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
    predictions = IsolationForest(n_estimators = 100).fit_predict(dist)
    
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
        print(f"Subsituting PSD {idx} with PSD {new_idx}")

    return psds

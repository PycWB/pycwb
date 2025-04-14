import numpy as np 
import ROOT
import logging
from pycwb.modules.cwb_conversions import *
from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.types.wdm import WDM
from functools import partial
import multiprocessing as mp
from memspectrum import MESA
import os 
from scipy import signal
from pycbc.filter.resample import highpass
from sklearn.ensemble import IsolationForest

#from pycbc.filter import highpass
os.environ['HOME_WAT_FILTERS'] = '/home/waveburst/SOFT/cWB/tags/config/O4_cWB_2G_config_v1.14/XTALKS'

def whitening_mesa(config, h):
    #Initialise WDM 
    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision)
    wdmFlen = wdm_white.m_H / config.rateANA

    #Sampling variables 
    sampling_rate = config.inRate / (2 ** config.levelR)
    Ny = .5 * sampling_rate #The Nyquist frequency 
    wdm_df = Ny / layers_white #frequency resolution for the WDM 
    wdm_dt = .5 / wdm_df #time resolution for the WDM 
    

    #Convert 
    h = h.to_pycbc() if type(h) == TimeSeries else h 
    h -= np.mean(h) 
    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high

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
    #Loop over data segment chunks 
    for i in range(n_windows + 1):
        start = i * stride
        M.solve(h[start:start+ window], method = config.mesaSolver, m = config.mesaOrder)
        f, psd = M.spectrum(1 / sampling_rate) 
        psds.append(psd) 
    
    #reindex psds to exclude possibly-glitch-contaminated estimates 
    outliers = np.repeat(False, len(psds)
    if config.mesaReindex: 
        psds, outliers = reindex_psds(np.array(psds), f)
    
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

    #Compute whitening factor in TF domain from time going to stride to stop - stride 
    nRMS_matrix = WSeries_to_matrix(tf_map) / WSeries_to_matrix(tf_map_white)
    

    #Compute the nRMS taking the median over 20 second segments and convert to array 
    data_per_batch = int(config.whiteStride // wdm_dt)
    nRMS_reshaped = nRMS_matrix.reshape(int(Ny / wdm_df),-1,data_per_batch) 
    nRMS_reshaped[:int(16 / wdm_df)] = 1  #set nRMS = 1 for f < 16 
    nRMS = np.sqrt(np.median(nRMS_reshaped ** 2, axis = 2))
    #nRMS[: int(16 / wdm_df)+1] = 1 #set nRMS = 1 for f < 16
    #add 1 as last frequency bin for compatibility 
    n_segments = nRMS_reshaped.shape[1]
    nRMS = np.vstack((nRMS,np.ones((1,n_segments)))).reshape(-1, order = 'F')  
    
 
    #convert to pycwb types nRMS
    nRMS = _convert_numpy_nrms_to_wseries(nRMS, h.start_time, sampling_rate, n_segments,  wdm_white.wavelet)
    tf_map_whitened = ROOT.WSeries(np.double)(convert_to_wavearray(h_white[:stop]), wdm_white.wavelet)
    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map_whitened)
    n_rms = convert_wseries_to_time_frequency_series(nRMS)

    return tf_map_whitened, n_rms, psds, outliers


def reindex_psds(psds, f): 

    #Compute ratio of PSDs over median over some frequencies bin to have a reference PSD 
    mask_lf = (f > 16) & (f < 128)
    mask_hf = (f > 128)
    psd_median = np.median(psds, axis = 0)

    #Compute the log distance over low and high frequency ranges to find IF features  
    dist_lf = np.mean((np.log(psds[:,mask_lf] / psd_median[mask_lf])) ** 2, axis = 1) ** .5
    dist_hf = np.mean((np.log(psds[:,mask_hf] / psd_median[mask_hf])) ** 2, axis = 1) ** .5
    dist = np.stack([dist_lf, dist_hf], axis = 1)
    predictions = IsolationForest(n_estimators = 100).fit_predict(dist)
    #Outliers are only above median 
    median_lf, median_hf = np.median([dist_lf,dist_hf], axis = 1)
    above_median = (dist_lf > median_lf) | (dist_hf > median_hf)
    predictions = np.logical_and(predictions == -1,above_median)
    #Reindex outliers 
    outliers_idx = np.where(predictions)[0] 
    inliers_idx = np.where(~predictions)[0] 

    for idx in outliers_idx: 
        new_idx = inliers_idx[np.argmin(np.abs(inliers_idx-idx))]
        psds[idx] = psds[new_idx]
        print(f"Subsituting PSD {idx} with PSD {new_idx}")
    return psds, predictions



def _convert_numpy_nrms_to_wseries(data: np.array, start: np.double, rate: int, n_segments: int,  wavelet):
    """
    Convert numpy array to wavearray with python loop

    :param data: numpy array
    :type data: np.array
    :param start: start time
    :type start: np.double
    :param stop: stop time
    :type stop: np.double
    :param rate: sample rate
    :type rate: int
    :return: Converted ROOT.wavearray
    :rtype: ROOT.wavearray
    """
    #create a wavearray containig zeros only. Do not append otherwise the size will double
                                   #len frequencies * n_segments 
    a = ROOT.wavearray(np.double)(int((len(data) - 2 * n_segments) // 2))
    ###This PROBABLY has to become the following: 
    #a = root.wavearray(np.double)((len(data) - n_segments) // 2 ) # how to insert n segments? 
    #this works! 60 is twice n segments. Two bins are splitted in 2 by the WDM. Why do we have to divide by two? Uhm... 
    #also len(data) // 2 - 60 works... Uhm... I am a bit confused. Are these the 90 and 0 degree phases? 
    #Can we Force length? 
    w = ROOT.WSeries(np.double)(a, wavelet) 
    w.Forward() 
    for i in range(len(data)):
        w.data[i] = data[i]
    #check if the data are right 
    w.start(start) 
    w.rate(rate)
    w.wrate(rate) # ----- THIS WAS COPIED BY cWB DEFINED QUANTITIES 
    w.f_high = rate / 2 
    w.pWavelet.allocate(w.size(),
                        w.data)
    #w_tfseries = convert_wseries_to_time_frequency_series(w) shold not stay there 



    return w

def convert_wavearray_to_wseries(data):
    """
    This is to convert wavearray to wseries, it substituted the wseries.Forward(wavearray) that is not working.
    Maybe we can understand why

    :param data: ROOT.wavearray
    :return: Converted ROOT.WSeries
    :rtype: ROOT.WSeries
    """
    w = ROOT.WSeries(np.double)()

    for d in data:
        w.append(d)

    w.start(data.start())
    w.rate(data.rate())
    w.wrate(0.)
    w.f_high = data.rate() / 2.
    w.pWavelet.allocate(w.size(),
                        w.data)
    return w

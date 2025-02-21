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

os.environ['HOME_WAT_FILTERS'] = '/home/waveburst/SOFT/cWB/tags/config/O4_cWB_2G_config_v1.14/XTALKS'



def whitening(config, h): 
    #convert to pycbc series to ensure compatibility 
    h = h.to_pycbc() if type(h) == TimeSeries else h

    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision) 
    wdmFlen = wdm_white.m_H / config.rateANA 
    if wdmFlen > config.segEdge + 0.001:
        logger.error("Error - filter scratch must be <= cwb scratch!!!")
        logger.error(f"filter length : {wdmFlen} sec")
        logger.error(f"cwb   scratch : {config.segEdge} sec")
        raise ValueError("Filter scratch must be <= cwb scratch!!!")
    else:
        logger.info(f"WDM filter max length = {wdmFlen} (sec)")

    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(h), wdm_white.wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)

    
    nRMS, tf_map_whitened = _whiten(config, h, tf_map, wdm_white.wavelet) 

    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map_whitened) 
    n_rms = convert_wseries_to_time_frequency_series(nRMS) 

    return tf_map_whitened, n_rms  


def _whiten(config,h,tf_map,wavelet):

    #Define time series sampling values 
    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    sampling_rate = config.inRate / (2 ** config.levelR) 
    Ny = .5 * sampling_rate 
    wdm_df = Ny / layers_white 
    wdm_dt = .5 / wdm_df
    #FOLLOWING ARE WRONG
    #wdm_dt = layers_white / 2 ** 12 #wdm time resolution 
    #wdm_df = .5 / wdm_dt  #wdm frequency resolution
    #Ny = int(.5 / h.delta_t) #Nyquist frequency
    #print(f'Samplng Rate: {sampling_rate}\n, Ny: {Ny}\ndf:{wdm_df}\ndt{wdm_dt}')

    n_segments = int(h.data.size // (config.whiteStride * sampling_rate)) #number of segments to be processed separately

    
    if config.whiteMethod == 'wavelet': 
        
    ######################
    #IMPLEMENTS CWB2G WHITENING 
    ######################
        nRMS = tf_map.white(config.whiteWindow, 0, config.segEdge,
                            config.whiteStride) 
        # high pass filtering at 16Hz
        nRMS.bandpass(16., 0., 1)
        # whiten  0 phase and 90 phase WSeries
        tf_map.white(nRMS, 1)
        tf_map.white(nRMS, -1)    
        wtmp = ROOT.WSeries(np.double)(tf_map)
        
        # average 00 and 90 phase
        tf_map.Inverse()
        wtmp.Inverse(-2)
        tf_map += wtmp
        tf_map *= 0.5

    elif config.whiteMethod == 'mesa': 
        
    ######################
    #IMPLEMENTS WHITENING WITH MESA 
    ######################        
        #Instatiate MESA and compute the power spectral density 
        h -= np.mean(h) 
        #Solve recursion and compute amplitude spectral density 
        M = MESA() 
        M.solve(h, method = config.whiteSolver, m = config.whiteOrder) 
        frequency, psd = M.spectrum(1 / sampling_rate, onesided = False)
        highpass = 16
        psd[(frequency <= highpass) & (frequency >= - highpass)] = 1
        h_white = (h.to_frequencyseries() / psd[:len(h) // 2 + 1]**0.5).to_timeseries() * np.sqrt(1 / sampling_rate)

        #Defne TF map for whitened data         
        tf_map_white = ROOT.WSeries(np.double)(convert_to_wavearray(h_white), wavelet)
        #without the forward the matrix at the end is empty 
        tf_map_white.Forward() 
        tf_map_white.setlow(config.fLow)
        tf_map_white.sethigh(config.fHigh)
        
        #Recover nRMS as root median square from whitened and non-whitened data 
        data_per_batch = int(config.whiteStride // wdm_dt)
        nRMS_matrix = WSeries_to_matrix(tf_map) / WSeries_to_matrix(tf_map_white)  
        cutoff = int(nRMS_matrix.shape[1] % (20 / wdm_dt))
                
        #check the length is proper to flatten nRMS array afterwards 
        if cutoff == 0: 
            pass
        else: 
            print('cutting last {} s of data'.format(cutoff * wdm_dt))
            nRMS_matrix = nRMS_matrix[:,:-cutoff]
        
        nRMS_reshaped = nRMS_matrix.reshape(int(Ny / wdm_df),-1,data_per_batch)
        nRMS = np.sqrt(np.median(nRMS_reshaped ** 2, axis = 2)).reshape(-1, order = 'F')          
        
        #convert to wseries to return them 
        nRMS = _convert_numpy_nrms_to_wseries(nRMS, h.start_time, sampling_rate, n_segments,  wavelet)
        tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(h_white), wavelet)
  
        

    elif config.whiteMethod == 'mixed':
          
    ######################
    #IMPLEMENTS WHITENING USING MESA IN WAVELET DOMAIN 
    ######################   
        
        #compute the nRMS at the given resolution level 
        compute_nrms_partial = partial(compute_nrms, config, h - np.mean(h), sampling_rate, n_segments, wdm_df, Ny)
        with mp.Pool(5) as p: 
            nRMS = p.map(compute_nrms_partial, range(n_segments))
        
        nRMS = np.reshape(nRMS, -1)
        nRMS = _convert_numpy_nrms_to_wseries(nRMS, h.start_time, sampling_rate, n_segments,  wavelet)
        nRMS.bandpass(16.,0.,1)
        # whiten  0 phase WSeries
        tf_map.white(nRMS, 1)
        # whiten 90 phase WSeries
        tf_map.white(nRMS, -1)
         
        wtmp = ROOT.WSeries(np.double)(tf_map)
        
        # average 00 and 90 phase
        tf_map.Inverse()
        wtmp.Inverse(-2)
        tf_map += wtmp
        tf_map *= 0.5

    else: 
        raise ValueError(f'The method {config.whiteMethod} is not available.')

    return nRMS, tf_map 
    
        
            

    

def compute_nrms(config, h, sampling_rate, n_segments, frequency_resolution, nyquist_frequency, i): 
            
        window = config.whiteWindow * sampling_rate
        stride = config.whiteStride * sampling_rate
    
        #Define start and stop for the method. Can it be simplified if jobs uses the scratch? I hope so 
        if i == 0: 
            start, stop = 0, 2 * stride
        elif i == n_segments - 1: 
            start, stop = (i-1) * stride, (i+1) * stride
        else: 
            start = (i-1) * stride
            stop = start + window
        
        length = int(stop - start) // 2 + 1 
        step = int(length * frequency_resolution / nyquist_frequency) 
        #Solve the Levinson recursion and compute the amplitude spectral density 
        M = MESA() 
        M.solve(h[int(start):int(stop)], method = config.whiteSolver, m = config.whiteOrder)
        asd = np.sqrt(M.spectrum(onesided = False)[1][:length])
        
        return asd[::step]     

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

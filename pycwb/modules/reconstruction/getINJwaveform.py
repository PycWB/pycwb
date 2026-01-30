import logging
import numpy as np
from pycwb.modules.cwb_conversions import convert_to_wseries, WSeries_to_matrix, convert_to_wavearray, convert_wavearray_to_timeseries
from pycwb.types.wdm import WDM

def get_INJ_waveform(hot, tf_map, gps_time, window, offset, in_rate) -> list:
    """
    get INJ waveforms and calculate injection statistics
    - estimate central time based on energy distribution (central time)
    - estimate injected waveforms 
        - apply window in time domain to contain 99.9% of the total energy 
        - apply frequency cut for search frequency band
    - calculate injection statistics (e.g., injected SNR, duration, central frequency, bandwidth, injected hrss)


    Parameters
    ----------
    hot : list of pycbc.types.timeseries.TimeSeries
        list of  pycbc.types.timeseries.TimeSeries objects
    tf_map : list of pycwb.types.wdm.WDM
        list of WDM objects
    gps_times : gps_time
        float
    window : half of the time window to contain the waveform
        float
    # offset : float
    #     type of waveforms, the value can be 'signal' or 'strain'
    in_rate : float
        sample rate of the original input data

    Returns
    -------
    output: list of dict
        list of dictionaries containing injection statistics and waveforms
    """
    
    hot = hot.data                              # original timeseries
    w = tf_map.data                             # whitened timeseries
    tf_map = convert_to_wseries(tf_map)         # whitened wseries
    tf_map.Forward()

    # seg_start = w.span[0] + offset + 1.         # analysed start time
    # seg_end = w.span[1] - offset - 1.           # analysed end time
    seg_start = float(w.start_time) + offset + 1.         # analysed start time
    seg_end = float(w.end_time) - offset - 1.           # analysed end time

    # outputs = []
    # loop over injections
    # for k in range(len(gps_times)):
        # central_time = gps_times[k]             # initial estimation on central_time
    
    # update central time
    t_start = max(float(w.start_time), gps_time - window)
    t_stop = min(gps_time + window, float(w.end_time))
    windowed_w = w.time_slice(t_start, t_stop)
    central_time = estimate_central_time(windowed_w)
    
    t_start = max(float(w.start_time), central_time - window)
    t_stop = min(central_time + window, float(w.end_time))
    windowed_w = w.time_slice(t_start, t_stop)
    # snr = estimate_snr(windowed_w)
    central_time = estimate_central_time(windowed_w)
    # duration = estimate_duration(windowed_w)

    # save None values if estimated central_time is outside of the segment
    if (central_time < seg_start) or (central_time > seg_end):
        return {'snr': None, 'central_time': None, 'duration': None,
                'central_freq': None, 'bandwidth': None, 'hrss': None,
                'whitened_injected_waveform': None, 'injected_strain': None}
    
    # hrss
    windowed_hot = hot.time_slice(t_start, t_stop)
    hrss = calculate_hrss(windowed_hot, in_rate)     # injected hrss

    # calculate central frequency and bandwidth in wavelet domain
    central_freq, bandwidth = estimate_central_frequency(tf_map, t_start, t_stop) # central_frequency, bandwidth

    # estimate final waveform
    # FIXME: add padding
    windowed_w = w.time_slice(central_time - window, central_time + window)
    waveform = estimate_waveform(windowed_w)
    waveform = apply_frequency_cut(waveform, tf_map.f_low, tf_map.f_high)

    # rescale data when data are resampled (resample with wavelet change the amplitude)
    rescale = 1. / np.sqrt(2) ** (np.log2(in_rate / waveform.sample_rate))
    waveform.data *= rescale

    # estimate statistics
    snr = estimate_snr(waveform)                    # iSNR
    central_time = estimate_central_time(waveform)  # central time
    duration = estimate_duration(waveform)          # duration

    # estimate injected strain
    # FIXME: add padding
    central_time2 = estimate_central_time(windowed_hot)

    windowed_hot = hot.time_slice(central_time2 - window, central_time2 + window)
    strain_waveform = estimate_waveform(windowed_hot)

    return  {'snr': snr, 'central_time': central_time, 'duration': duration,
             'central_freq': central_freq, 'bandwidth': bandwidth, 'hrss': hrss,
             'whitened_injected_waveform': waveform, 'injected_strain': strain_waveform}

def estimate_snr(h1, h2 = None):
    """
    calculate power snr of the given waveform
    """
    if h2 is None:
        return np.sum(h1.data * h1.data)
    else:
        # window for overlapping segment
        t_start = max(h1.start_time, h2.start_time)
        t_stop = min(h1.end_time, h2.end_time)
        
        windowed_h1 = h1.time_slice(t_start, t_stop)
        windowed_h2 = h2.time_slice(t_start, t_stop)

        if len(windowed_h1) != len(windowed_h2):
            num_points = min(len(windowed_h1), len(windowed_h2))
            windowed_h1.data = windowed_h1.data[:num_points]
            windowed_h2.data = windowed_h2.data[:num_points]
        
        return np.sum(windowed_h1.data * windowed_h2.data)

def estimate_central_time(h):
    return np.sum(h.data * h.data * h.sample_times.data) / np.sum(h.data * h.data)

def estimate_duration(h):
    central_time = estimate_central_time(h)
    rel_time = h.sample_times.data - central_time
    return np.sqrt(np.sum(h.data * rel_time * h.data * rel_time) / np.sum(h.data * h.data))

def calculate_hrss(h, inRate):
    # FIXME: pD->rate = 16384. is the original sample rate before downsampling ...
    return np.sqrt(np.sum(h.data * h.data) / inRate)

def estimate_central_frequency(tf_map, t_start, t_stop):
    # convert time window to index window
    idx_t_start = int((t_start - tf_map.start()) * tf_map.wRate)
    idx_t_stop = int((t_stop - tf_map.start()) * tf_map.wRate)
    zero_layer = tf_map.getSlice(0)
    if idx_t_start<=0: idx_t_start=0
    if idx_t_stop>=int(zero_layer.size()): idx_t_stop = zero_layer.size() - 1
    
    mx_map = WSeries_to_matrix(tf_map)
    # tf_times = tf_map.start() + np.arange(mx_map.shape[1]) / tf_map.wRate
    tf_freqs = np.array([tf_map.frequency(l) for l in range(tf_map.maxLayer()+1)])

    # apply window
    mx_map = mx_map[:, idx_t_start:idx_t_stop]
    
    rms = np.std(mx_map, axis=1)
    central_freq = np.sum(tf_freqs * rms * rms) / np.sum(rms * rms)

    bandwidth = np.sum(tf_freqs * tf_freqs * rms * rms) / np.sum(rms * rms) - central_freq * central_freq
    bandwidth = np.sqrt(bandwidth) if bandwidth>0 else 0.

    return central_freq, bandwidth

def estimate_waveform(h):
    """
    estimate windowed waveform that contains 99.9% of the entire energy threshold
    """
    
    idx_center = len(h.data) // 2
    left = h.data[0:idx_center][::-1]
    right = h.data[idx_center+1:]
    left = np.pad(left, (0, max(len(left), len(right)) - len(left)))
    right = np.pad(right, (0, max(len(left), len(right)) - len(right)))

    sum_E = left * left + right * right
    cum_E = np.cumsum(np.insert(sum_E, 0, h.data[idx_center] * h.data[idx_center]))
    
    # Fallback: if 99.9% is never reached (e.g. signal is effectively noise), use full width
    idx = np.where(cum_E > (0.999 * estimate_snr(h)))[0]
    idx = idx[0] if len(idx)>0 else min(len(left), len(right))
    idx = max(1, idx) # Enforce min value (C++ loop starts at j=1)

    t_start = h.sample_times.data[idx_center] - idx / h.sample_rate
    t_stop = h.sample_times.data[idx_center] + idx / h.sample_rate
    waveform = h.time_slice(t_start, t_stop)
    
    return waveform

def apply_frequency_cut(h, f_low, f_high):
    w = convert_to_wavearray(h)
    w.FFTW(1)

    fs = w.rate() / w.size() / 2
    # FIXME: it would be nice to vectorize but wavearray.data[i] only takes single value ...
    # FIXME: Or pythonize?
    for j in range(0, w.size()//2, 2):
        f = j * fs
        
        if f < f_low or f > f_high:
            w.data[j] = 0.0   
            w.data[j+1] = 0.0

    w.FFTW(-1)

    # convert to pycbc for consistency
    waveform = convert_wavearray_to_timeseries(w).to_pycbc()
    
    return waveform
from pycbc.types.timeseries import load_timeseries, TimeSeries 
import numpy as np 
from scipy.signal import correlate, hilbert 
import cmath 


class Waveform(TimeSeries): 
    """
    Class to handle waveform data.

    :param data: data
    :type data: pycbc.types.timeseries.TimeSeries
    """
    def __init__(self, data: TimeSeries):
        super().__init__(data, data._delta_t, data.start_time)
        self._findStartEnd() 
        #Store time and phase shift to reverse the synchronization
        self._total_time_shift = 0 
        self._total_phase_shift = 0


    def estimateCentralTime(self): 
        """
        Estimate the central time of the waveform as the weighted average of sample times, weighted by the signal energy.
        """
        central_time = np.sum(self.data * self.data * self.sample_times.data) / np.sum(self.data * self.data)
        self.central_time = central_time 
        return central_time 

    def estimateDuration(self, sigma = 3):
        """
        Estimate the duration of the wavefrom computing time variance around central time. The total duration is then sigma times this value.
        Standard is 3sigma (~99.7% of the signal energy).
        :param sigma: Multiplier for the duration estimate. Default is 3.
        """
        central_time = self.estimateCentralTime()
        rel_time = self.sample_times.data - central_time
        duration = np.sqrt(np.sum(self.data * rel_time * self.data * rel_time) / np.sum(self.data * self.data)) * sigma
        self.signalDuration = duration 
        return duration

    def _findStartEnd(self): 
        """
        Find the start and end of the waveform, as indeces (istart, iend) and times (tstart, tend)
        and creates relative attributes in the waveform object.
        """
        central_time = self.estimateCentralTime()
        duration = max(self.estimateDuration(), 0.150) # Ensure minimum duration of 0.1s to avoid too narrow windows

        self.tstart, self.tend = central_time - duration / 2, central_time + duration / 2 
        self.istart = int(round((self.tstart - self.sample_times.data[0]) / self.delta_t))
        self.iend   = int(round((self.tend - self.sample_times.data[0]) / self.delta_t))
        

    def syncWaveform(self, reference_waveform, sync_phase = True): 
        """
        Synchronize this waveform with a reference waveform.

        :param reference_waveform: The reference waveform to synchronize with.
        :type reference_waveform: Waveform
        :param sync_phase: Whether to synchronize the phase as well. Default is True.
        :type sync_phase: bool
        """
        #
        if not isinstance(reference_waveform, Waveform):
            raise ValueError("Reference waveform must be an instance of Waveform.")
        
        if self.sample_rate != reference_waveform.sample_rate:
            raise ValueError("Sample rates do not match.")

        self.timeShift(self.computeTimeDifference(reference_waveform)) 
        reference_padded = self.pad_to_same_length(reference_waveform)
        if sync_phase:
            self.data = self.phaseShift(self.computePhaseDifference(reference_padded))


    def retriveOnsourceTimes(self): 
        """
        Retireve the original on-source time of the waveform, re-establishing original
        """
        self.data = self.phaseShift(-self._total_phase_shift)
        self.timeShift(-self._total_time_shift)


    def fft(self, direct = True): 
        """
        Compute the FFT of the waveform data.

        :param direct: If True, compute the direct FFT. If False, compute the inverse FFT.
        :type direct: bool
        :return: The waveform with FFT applied.
        :rtype: Waveform
        """
        
        #Compute direct FFT and store frequencies in attribute
        if direct: 
            if hasattr(self, 'sample_frequencies'):
                "Direct FFT is already computed."
                return 
            self.frequencies = np.fft.rfftfreq(len(self.data), d=self.delta_t)
            self.data = np.fft.rfft(self.data, norm = 'ortho')

        #Compute inverse FFT. 
        if not direct: 
            if not hasattr(self, 'sample_frequencies'):
                raise ValueError("Direct FFT not computed. Call fft(direct=True) first.")
            self.data = np.fft.irfft(self.data, norm = 'ortho')
            del(self.sample_frequencies)

        return self 


    def pad_to_same_length(self, reference_waveform):
        """
        Pad this waveform and a reference waveform with zeros so they share
        the same start and end times. Returns both padded waveforms.

        :param reference_waveform: Waveform to align with
        :return: (padded_self, padded_reference)
        """
            # Make copies so we don’t modify originals
        ref_pad = reference_waveform.copy()
        dt = self.delta_t

        # Determine combined start and end times
        t_start = min(self.sample_times[0], ref_pad.sample_times[0])
        t_end   = max(self.sample_times[-1],  ref_pad.sample_times[-1])

        # Pad self waveform
        n_pre_w  = int(round((self.sample_times[0] - t_start) / dt))
        n_post_w = int(round((t_end - self.sample_times[-1]) / dt))
        if n_pre_w > 0:
            self.prepend_zeros(n_pre_w)
        if n_post_w > 0:
            self.append_zeros(n_post_w)
        # Pad reference waveform
        n_pre_r  = int(round((ref_pad.sample_times[0] - t_start) / dt))
        n_post_r = int(round((t_end - ref_pad.sample_times[-1]) / dt))
        if n_pre_r > 0:
            ref_pad.prepend_zeros(n_pre_r)
        if n_post_r > 0:
            ref_pad.append_zeros(n_post_r)

        return ref_pad

    def _computeCrossCorrelation(self, reference_waveform):
        """
        Compute the cross-correlation between this waveform and another waveform.
        """
        
        # length of signals
        n1 = len(reference_waveform.data)
        n2 = len(self.data)

        dt = 0 
        if reference_waveform.start_time != self.start_time:
            dt = float(reference_waveform.start_time - self.start_time) 

        # correct lag array: from -(n2-1) to (n1-1)
        lags = np.arange(-(n2-1), n1) / self.sample_rate + dt 

        return lags, correlate(reference_waveform.data, self.data, method='fft', mode='full')


    def computeTimeDifference(self, reference_waveform):
        """
        Compute the time difference between this waveform and another waveform using cross-correlation.
        """

        lags, cc = self._computeCrossCorrelation(reference_waveform)
        max_index = cc.argmax()
        time_shift = lags[max_index]

        return time_shift


    def timeShift(self, shift): 
        """
        Shift the waveform in time by a specified amount.
        """
        if shift == 0: 
            return 
        self.start_time += shift 
        self._total_time_shift += shift
        #updated start, end indeces after time shifting 
        self._findStartEnd() 


    def computePhaseDifference(self, reference_waveform):
        """
        Compute phase difference using samples starting from the maximum start time
        and taking the maximum possible common length.
        """

        w1 = self.data
        w2 = reference_waveform.data

        # 90-degree phase-shifted versions
        w1_90 = hilbert(w1)
        w2_90 = hilbert(w2)

        # Phase difference estimator
        num = np.sum(w1 * w2_90 - w1_90 * w2) 
        den = np.sum(w1 * w2 + w1_90 * w2_90)

        # Deterministic fallback (never zero output)
        if num == 0 and den == 0:
            num = np.finfo(float).eps

        return -np.arctan2(np.real(num), np.real(den)) 




    def phaseShift(self, shift):
        """
        Shift the phase of the waveform by a specified amount.
        """
        #Raise error if shift is not a number 
        if not isinstance(shift, (int, float)):
            raise ValueError("Shift must be a numeric value.")

        #If shift is zero, do nothing        
        if shift == 0: return self.data 
        #Apply phase shift to the waveform data
        data_fft = np.fft.rfft(self.data)
        #Apply the phase shift to the FFT data
        data_fft_shifted = data_fft * cmath.rect(1, shift)
        #Store in the total phase shift
        self._total_phase_shift += shift
        #Convert back to time domain
        return np.fft.irfft(data_fft_shifted, n = len(self.data)) 
        


    def resample(self, new_sample_rate):
        """
        Resample the waveform to a new sample rate.
        """
        if not isinstance(new_sample_rate, (int, float)) or new_sample_rate <= 0:
            raise ValueError("New sample rate must be a positive numeric value.")
        
        resampled_data = self.resample(1 / new_sample_rate)
        return self.__class__(resampled_data)



    def overlap(self, other_waveform): 
        """
        Compute the overlap between this waveform and another waveform.
        """
        if not isinstance(other_waveform, Waveform):
            raise ValueError("Can only match with another waveform instance.")
        
        # Check if the data arrays are equal
        num = np.sum(self.data * other_waveform.data)
        den = np.sqrt(np.sum(self.data**2) * np.sum(other_waveform.data**2))
        return num / den 


    def RMS(self, norm = False):
        """ 
        Return the RMS value of the waveform.

        :param norm: If True, normalize by the number of samples.
        :type norm: bool
        :return: RMS value
        :rtype: float
        """
        squareSum = np.sum(self.data[self.istart:self.iend]**2)

        if norm:
            squareSum /= self.data.size 

        return np.sqrt(squareSum) 


    def rollingRMS(self, norm = False): 
        """
        Return the rolling RMS of the waveform.

        :param norm: If True, normalize by the number of samples.
        :type norm: bool
        :return: rolling RMS array
        :rtype: np.ndarray
        """
        cumsum = np.cumsum(self.data[self.istart:self.iend] ** 2) 
        if norm: 
            cumsum /= self.data.size
        return np.sqrt(cumsum)    

    def max(self): 
        """
        Return the maximum value of the waveform.
        """
        return self.data.max() 


    def argmax(self): 
        """
        Return the index of the maximum value of the waveform.
        """
        return self.data.argmax() 


    def copy(self): 
        """
        Return a copy of the waveform.
        """
        copy = self.__class__(self) 
        copy._total_time_shift = self._total_time_shift
        copy._total_phase_shift = self._total_phase_shift
        return self.__class__(self)

    def __len__(self):
        return len(self.data)


def load_waveform(filename, resample = None):
    """
    Load a waveform from a file.
    """
    # Assuming the waveform is stored in a file, you can use pycbc's load_timeseries
    # to load the waveform data.
    data = load_timeseries(filename)
    if isinstance(resample, int) or isinstance(resample, float):
        data = data.resample(resample)
    return Waveform(data) 






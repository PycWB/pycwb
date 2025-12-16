from pycbc.types.timeseries import load_timeseries, TimeSeries 
import numpy as np 
from scipy.signal import correlate 
import cmath 


class Waveform(TimeSeries): 
    """
    Class to handle waveform data.

    :param data: data
    :type data: pycbc.types.timeseries.TimeSeries
    """
    def __init__(self, data: TimeSeries):
        super().__init__(data, data._delta_t, data.start_time)
        self.findStartEnd() 
        #Store time and phase shift to reverse the synchronization
        self._total_time_shift = 0 
        self._total_phase_shift = 0


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
        
        self.alignStartTime(reference_waveform)
        self.timeShift(self.computeTimeDifference(reference_waveform)) 
        if sync_phase:
            self.data = self.phaseShift(self.computePhaseDifference(reference_waveform))


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


    def alignStartTime(self, reference_waveform): 
        """
        Align the start time of this waveform with a reference waveform.

        :param reference_waveform: The reference waveform to align with.
        :type reference_waveform: Waveform
        """

        time_shift = reference_waveform.start_time - self.start_time
        if time_shift != 0:
            self.timeShift(time_shift)

    
    def findStartEnd(self, rtol=1e-3):
        """
        Find start and end of the waveform as the contiguous segment
        containing the global maximum above threshold.
        """
        threshold = np.max(np.abs(self.data)) * rtol

        peak_idx = np.argmax(np.abs(self.data))

        # Expand backwards
        istart = peak_idx
        while istart > 0 and np.abs(self.data[istart]) >= threshold:
            istart -= 1

        # Expand forwards
        iend = peak_idx
        while iend < len(self.data) - 1 and np.abs(self.data[iend]) >= threshold:
            iend += 1

        self.istart = istart
        self.iend = iend
        self.tstart = self.sample_times[istart]
        self.tend = self.sample_times[iend]

        return self.tstart, self.tend

    def findStartEnd_OLD(self, rtol = 1e-3): 
        """
        Find the start and end of the waveform, as indeces (istart, iend) and times (tstart, tend)
        and creates relative attributes in the waveform object.

        :param rtol: relative tolerance to define non-zero values
        :type rtol: float
        :return: (tstart, tend)
        :rtype: tuple containing the estimated start and end times
        """
        non_zero_indices = np.where(np.abs(self.data) >= self.data.max() * rtol)[0]
        if non_zero_indices.size == 0:
            raise ValueError("Waveform data is all zeros or below the threshold.")
        self.istart, self.iend = non_zero_indices[0], non_zero_indices[-1]
        self.tstart, self.tend = self.sample_times[self.istart], self.sample_times[self.iend]
        return self.tstart, self.tend     



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
        if shift == 0: return 
        self.start_time += shift 
        self._total_time_shift += shift
        #updated start, end indeces after time shifting 
        self.findStartEnd() 


    def computePhaseDifference(self, reference_waveform): 
        """
        Synchronize the phase of this waveform with another waveform.
        """
        #Reinitialize the waveforms to the same time slice
        start, end = max(self.tstart, reference_waveform.tstart), min(self.tend, reference_waveform.tend)
        this = self.__class__(self.time_slice(start, end)) 
        reference = self.__class__(reference_waveform.time_slice(start,end))
        #Compute 90_degree phase shifted versions of the waveforms
        this_90 = this.phaseShift(np.pi / 2)
        reference_90 = reference.phaseShift(np.pi / 2) 

        num, den = 0, 0
        for i in range(np.size(this)):
            try: 
                num += this[i] * reference_90[i] - this_90[i] * reference[i]
                den += this[i] * reference[i] + this_90[i] * reference_90[i]
            except IndexError: 
                pass 
        #Compute the phase difference with - sign to call "Phase Shift" without changing sign 
        phase_diff = - np.arctan2(num, den)
    
        return phase_diff   


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

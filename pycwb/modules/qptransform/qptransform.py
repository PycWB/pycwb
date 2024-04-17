"""
This code implements the wavelet Q-transform 
and its chirplet version, the wavelet Qp-transform, with Gaussian-wavelets with increasing (or eventually decreasing) frequency
Moreover, this code implements the possibility to invert the Qp-transform 
(and so the Q-transform as well) with a non standard inversion formula. 
This leads also to a denoising formula which allows to set to zero some (likely noisy) time-frequency regions
and to obtain the corresponding time series.
This code has been written starting from https://github.com/gwpy/gwpy/blob/main/gwpy/signal/qtransform.py
and even if many important parts have been changed with respect to the original code
the implementation below owes much to the GWpy authors, at least in the overall structure.
"""


import warnings
import sys
import numpy

import matplotlib.pyplot as plt
plt.rcParams['text.usetex']=True

from gwpy.utils import round_to_power
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram

import scipy
from scipy import integrate
from scipy import interpolate
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy import special

__author__ = 'Andrea Virtuoso <andrea.virtuoso@ts.infn.it>'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>, Scott Coughlin <scott.coughlin@ligo.org>, Alex Urban <alexander.urban@ligo.org>'

# Qp-transform defaults
DEFAULT_FRANGE = (0., float('inf'))
DEFAULT_QRANGE = (2.*numpy.pi, 20.*2.*numpy.pi)
DEFAULT_PRANGE = None
DEFAULT_ALPHA = 1.
DEFAULT_MAX_Qp_ITERS = 100
DEFAULT_MAX_BASINHOPPING_ITERS = 5
NUMBER_OF_SIGMAS = 3.
DEFAULT_ENERGY_DENSITY_THRESHOLD = 7.
DEFAULT_MINIMIZE_METHOD='SLSQP'#'Nelder-Mead'#

# -- object class definitions -------------------------------------------------

class QpObject(object):
    """
    Base class for Qp-transform objects
    This object exists just to provide basic methods for other Qp-transform objects.
    Parameters
	----------
    Q : the Q-value for this plane
    p : the p-value for this plane
    duration : the duration in seconds of the data
    sampling : sampling rate in Hz of the data
    alpha : the distance in units of Gaussian-sigmas between different Qp-transform points
        i.e. for a given time-frequency point (tau_0, phi_0) the nearest point in time is at tau_1=tau_0+alpha*sigma_tau(tau=tau_0),
        the nearest point in frequency is at phi_1=phi_0+alpha*sigma_phi(phi=phi_0)
    """
    def __init__(self, Q, p,  duration, sampling, alpha=DEFAULT_ALPHA):
        self.Q=Q
        self.p=p
        self.duration=duration
        self.sampling=sampling
        self.alpha=alpha



class QpTile(QpObject):
    """
    Representation of a tile with fixed Q, p and frequency
    Parameters (not inherited by QpObject)
	----------
    frequency: the central frequency associated with this tile
    """
    def __init__(self, Q, p, duration, sampling, frequency, alpha=DEFAULT_ALPHA):
        super().__init__(Q, p,  duration, sampling, alpha)
        
        self.frequency=frequency    
	  
    @property
    def sigma_tau(self):
        return self.Q/(4.*numpy.pi*self.frequency)

    @property
    def sigma_phi(self):
        return (self.frequency/self.Q)*numpy.sqrt(1+((2*self.p*self.Q)**2))

    @property
    def ntiles(self):
        """The number of tiles in this row (i.e. duration/(alpha*sigma_tau)), rounded to next power of 2 to speed up FFT
        If effective sampling (i.e. n_tiles/duration) is lower than 
        upper frequency covered by Gaussian window (i.e. frequency+NUMBER_OF_SIGMAS*sigma_phi)
        then 2*n_tiles is taken until a sufficient effective sampling is reached 
        """	
        n_tiles=round_to_power(self.duration/(self.alpha*self.sigma_tau), base=2, which='upper')
        while(0.5*(n_tiles/self.duration)<self.frequency+NUMBER_OF_SIGMAS*self.sigma_phi+1./self.duration): 
            n_tiles*=2
        return n_tiles

    @property
    def windowsize(self): 
        """The size of the frequency-domain window for this row 
        For Gaussian window it has been decided to integrate up to +-NUMBER_OF_SIGMAS*sigma_phi (default is 3.)
        """
        return 2 * int((NUMBER_OF_SIGMAS*self.sigma_phi) * self.duration)

    def _get_indices(self):
        half = int((self.windowsize) / 2)
        return numpy.arange(-half, half+1, 1, dtype=int) # last point is not included

    def get_data_indices(self):
        """Returns the index array of interesting frequencies for this row
		"""
        return numpy.round(self._get_indices() + self.frequency * self.duration).astype(int) 
	
    def get_window(self, fft_freqs): 
        """Generate the Gaussian window for this row with 'frequency' as central frequency
        """
        Qtilde=self.Q/numpy.sqrt(1.+2.j*self.Q*self.p)
        norm=((2.*numpy.pi/(self.Q**2))**(1./4.))*numpy.sqrt(1./(2.*numpy.pi*self.frequency))*Qtilde
        exp_argument=((0.5*Qtilde*(fft_freqs-self.frequency))/self.frequency)**2	
        return norm*numpy.exp(-exp_argument)

    @property
    def padding(self):
        """The padding required for the IFFT
        For how the following 'transform' works, padding is done at right 
        """
        pad = self.ntiles - self.windowsize 
        return (0, pad-1)

    def transform(self, fft_series, fft_freqs, start_time, derivative=False, original_sampling_rate=False): 
        """Calculate the energy `TimeSeries` for the given fseries
        Parameters
        ----------
        fft_series: the FFT of the data, obtained with scipy.fft.fft 
        fft_freqs: the frequencies associated with 'fft_series', obtained with scipy.fft.fftfreq
        start_time: the time to give to TimeSeries as x0
        derivative: if 'True' it calculates the derivative of the transform with respect to time (i.e. 'tau'), needed for reconstruction formula
        original_sampling_rate: if 'True' the transform is calculated at each frequency for a number of time points equal to that of the starting time-series
        Returns
        -------
        transform_ts: TimeSeries object with the result of the QpTransform for this tile; note that this is a complex number
        """

        # this is to tile time points according to time resolution at each frequency 
        if(original_sampling_rate==False):

            start_len=len(fft_freqs)
            fft_series=fft_series[self.get_data_indices()]
            fft_freqs=fft_freqs[self.get_data_indices()]

            if(derivative is True):
                fft_series*=2.*numpy.pi*1.j*fft_freqs

            windowed = fft_series * self.get_window(fft_freqs)	
            
            # this needs a freqeuncy shift (performed later)
            transform_fft=numpy.pad(windowed, self.padding, mode='constant')

            # normalise the FFT to the new number of elements (very important!)
            transform_fft*=(self.ntiles/(start_len))
            
            if(start_time is None):
                start_time=0.
            # IFFT and return a `TimeSeries`, normalising by sqrt(sampling)
            transform = scipy.fft.ifft(transform_fft, norm='backward')
            transform_ts = numpy.sqrt(self.sampling)*TimeSeries(transform, x0=start_time, dx=self.duration/transform.size, copy=False)
            
            # here we properly shift in frequency
            transform_ts*=numpy.exp(-2.j*numpy.pi*fft_freqs[0]*transform_ts.times.value)

        # this instead takes the same number of time points at each frequency (i.e. those given by the sampling rate); that is useful for inversion formula and spectrogram representation
        else:
          
            if(derivative is True):
                fft_series*=2.*numpy.pi*1.j*fft_freqs
            windowed = fft_series * self.get_window(fft_freqs)	

            if(start_time is None):
                start_time=0.

            transform = scipy.fft.ifft(windowed, norm='backward')
            transform_ts = numpy.sqrt(self.sampling)*TimeSeries(transform, x0=start_time, dx=self.duration/transform.size, copy=False)

        return transform_ts
			
	

class QpPlane(QpObject):
    """
    Qp-transform plane made by QpTile objects calculated for different frequencies at fixed Q and p
    """
    def __init__(self, frange, Q, p, duration, sampling, alpha=DEFAULT_ALPHA):
        super().__init__(Q, p,  duration, sampling, alpha)
        
        self.frange = [float(frange[0]), float(frange[-1])]

        lowest_phi=(1./self.duration)*(1.-(NUMBER_OF_SIGMAS*numpy.sqrt(1+((2*self.p*self.Q)**2)))/self.Q)
        highest_phi=(self.sampling / 2.)/(1.+(NUMBER_OF_SIGMAS*numpy.sqrt(1+((2*self.p*self.Q)**2)))/self.Q)  # no integration above sampling=2*(frange[-1]+(NUMBER_OF_SIGMAS*numpy.sqrt(1+((2*self.p*self.Q)**2)))*sigma_phi(phi=frange[-1]))

        
        if(self.frange[0] < lowest_phi):  # set non-zero lower frequency
            self.frange[0] = lowest_phi
            warnings.warn(f"lower bound of frequency range set to {lowest_phi} Hz")

        if(self.frange[-1] > highest_phi):  # set non-infinite upper frequency
            self.frange[-1] = highest_phi
            warnings.warn(f"upper bound of frequency range set to {highest_phi} Hz")

    @property
    def frequencies(self):
        """
        Given frange and alpha, this finds the frequencies on which the Qp-transform is calculated
        """
        nfreq = int(numpy.ceil(numpy.log(self.frange[1]/self.frange[0])/numpy.log(1.+(self.alpha*numpy.sqrt(1+((2*self.p*self.Q)**2)))/self.Q)))
        return numpy.geomspace(self.frange[0], self.frange[1], nfreq)
    


    def transform(self, fft_series, fft_freqs, start_time, energy_density_threshold=DEFAULT_ENERGY_DENSITY_THRESHOLD, derivative=False, original_sampling_rate=False):
        """
        Calculate both the time-frequency plane made by QpTile objects at each frequency of frequencies ('plane')
        and the energies associated with this plane ('energies')
        Parameters
        ----------
        fft_series : the FFT of the data, obtained with scipy.fft.fft 
        fft_freqs : the frequencies associated with 'fft_series', obtained with scipy.fft.fftfreq
        start_time : the time to give to TimeSeries as x0
        energy_density_threshold : the threshold above which the energy density of the plane is calculated
        derivative : if 'True' it calculates the derivative of the transform with respect to time (i.e. 'tau'), needed for reconstruction formula
        original_sampling_rate: if 'True' the transform is calculated at each frequency for a number of time points equal to that of the starting time-series
        Returns
        -------
        QpGram object
        """
        plane = []
        energies = []
        
        # for each frequency calculate tile and its energy
        for frequency in self.frequencies:
            tile=QpTile(Q=self.Q, p=self.p, duration=self.duration, sampling=self.sampling, frequency=frequency, alpha=DEFAULT_ALPHA).transform(fft_series=fft_series, fft_freqs=fft_freqs, start_time=start_time, derivative=derivative, original_sampling_rate=original_sampling_rate)
            plane.append(tile)
            energy_tile=TimeSeries(abs(tile.value)**2, times=tile.times, copy=False)
            energies.append(energy_tile)

        return QpGram(plane, energies, frequencies=self.frequencies, Q=self.Q, p=self.p, duration=self.duration, sampling=self.sampling, alpha=self.alpha, energy_density_threshold=energy_density_threshold)



class QpGram(QpObject):
    """
    Parameters (not inherited by QpObject)
	----------
    plane : the complex values obtained with the Qp-transform
    energies : the squared module of the Qp.transform (i.e. abs(plane)**2) 
    frequencies : the frequencies on which the Qp-transform is calculated
    energy_density_threshold : the threshold above which the energy density of the time-frequency map is calculated
    peak : energy of the loudest tile and the corresponding time and frequency
    energy_density : the energy of the time-frequency region for which energy>energy_density_threshold divided by the corresponding TF_area
    TF_area : the time-frequency area for which energy>energy_density_threshold
    """
    def __init__(self, plane, energies, frequencies, Q, p, duration, sampling, alpha, energy_density_threshold):
        super().__init__(Q, p, duration, sampling, alpha)
    
        self.plane=plane
        self.energies=energies
        self.frequencies=frequencies
        self.energy_density_threshold=energy_density_threshold
        self.weighted_energy_density_threshold=self.energy_density_threshold*numpy.average(self.frequency_weight())
        self.peak=self.find_peak()
        self.energy_density, self.TF_area=self.find_energy_density()
        self.weighted_energy_density=self.find_weighted_energy_density()
        
        
     
    def find_peak(self):
        """Find the energy of the loudest tile and the corresponding time and frequency
        """
        peak = {'energy': 0, 'time': None, 'frequency': None}
        item=0
        for energy in self.energies:
            maxidx = energy.value.argmax()
            maxe = energy.value[maxidx]
            if maxe > peak['energy']:
                peak.update({
                    'energy': maxe,
                    'time': energy.t0.value + energy.dt.value * maxidx,
                    'frequency': self.frequencies[item],
				})
            item+=1
        return peak
	
	
    def find_energy_density(self):
        """Find the energy density for the energy above the given energy_density_threshold
        """
        energy_integral_array=numpy.zeros(len(self.frequencies))
        area_integral_array=numpy.zeros(len(self.frequencies))
        item=0
        for energy in self.energies:
            energy=energy.value.copy()
            area=numpy.ones(len(energy))
            energy[energy<self.energy_density_threshold]=0. 
            area[energy<self.energy_density_threshold]=0.
            times=numpy.linspace(0., self.duration, len(energy))
            energy_integral_array[item]=integrate.trapezoid(energy, times) # use trapezoid for avoiding that in iterpolating zero and non zero it takes negative values
            area_integral_array[item]=integrate.trapezoid(area, times)
            item+=1
        energy_integral=integrate.trapezoid(energy_integral_array, self.frequencies) 
        area_integral=integrate.trapezoid(area_integral_array, self.frequencies)
        if(area_integral!=0.):
            return energy_integral/area_integral, area_integral 
        else:
            return 0., 0.
        
    def frequency_weight(self):
        return self.frequencies/self.frequencies[-1]#numpy.sqrt(self.frequencies/self.frequencies[-1])
    
    def find_weighted_energy_density(self):
        """Find the energy density weighted on frequency (this should focus more the choice of Q and p on merger)
        """
        energy_integral_array=numpy.zeros(len(self.frequencies))
        area_integral_array=numpy.zeros(len(self.frequencies))
        item=0
        weight=self.frequency_weight()
        for energy in self.energies:
            energy=energy.value.copy()
            area=numpy.ones(len(energy))
            energy[energy<self.weighted_energy_density_threshold]=0. 
            area[energy<self.weighted_energy_density_threshold]=0.
            times=numpy.linspace(0., self.duration, len(energy))
            energy_integral_array[item]=weight[item]*integrate.trapezoid(energy, times) # use trapezoid for avoiding that in iterpolating zero and non zero it takes negative values
            area_integral_array[item]=integrate.trapezoid(area, times)
            item+=1
        energy_integral=integrate.trapezoid(energy_integral_array, self.frequencies) 
        area_integral=integrate.trapezoid(area_integral_array, self.frequencies)
        if(area_integral!=0.):
            return energy_integral/area_integral
        else:
            return 0.



# main class
		
class QpTransform():
    """
    This class gives the Qp transform of a given series
    Inputs
    ----------
    data : the data needed for Qp-transform
    times : the time array corresponding to 'data'
    outseg : the specific time segment on which the Qp-transform will be applied; if 'None' no cut is applied
    whiten : if 'True' whitening is applied
    derivative : if 'True' the time derivative of the Qp-transform is calculated in 'qpgram_derivative'
    frange : the frequency range on which the Qp-transform is calculated
    alpha : the alpha used for calculate the final Qp-transform (i.e. once Q and p have been found)
    alpha_find_Qp : the alpha used for finding the Q and p values throug the maximization of the qpgram energy density;
        this is usually higher than 'alpha'
        IMPORTANT: number of time-frequency points (and so computational time) scales as 1/(alpha**2)
    energy_density_threshold: the threshold above which the energy density of the time-frequency map is calculated
    qrange : the selected range for finding the Q value which maximise the energy density
    prange : the selected range for finding the p value which maximise the energy density
    max_Qp_iters : max number of maximizer iterations for finding (Q,p)
    filmethod : method to filter the data. 
        If None, data are not filtered, if not None filtered data are returned as 'filseries'
        Filtering methods can be
            'highpass_threshold_filtering'->takes all time-frequency regions with energy above 'denoising_threshold'
            'lowpass_threshold_filtering'->takes all time-frequency regions with energy below 'denoising_threshold'
            'rangepass_threshold_filtering'->in that case 'denoising_threshold' should be made by two elements, 
            filmethod takes all time-frequency regions with energy between 'denoising_threshold[0]' and 'denoising_threshold[-1]'
    denoising_threshold : the thresold taken as reference for 'filmethod'
        if 'filmethod'=='rangepass_threshold_filtering' then 'denoising_threshold' should be made by two elements
    Parameters
    ----------
    series : TimeSeries object obtained from 'data', taking into account for 'outseg' and 'whiten'
    times : the time array corresponding to 'series' 
    qpgram : the QpGram object resulting from the Qp-Transform
    Q : the Q value for this transform
    p : the p value for this transform
    peak : energy of the loudest tile and the corresponding time and frequency
    energy_density : the energy of the time-frequency region for which energy>energy_density_threshold divided by the corresponding TF_area
    TF_area : the time-frequency area for which energy>energy_density_threshold
    qpgram_derivative : if 'derivative' is True this contains the derivative of the transform with respect to time (i.e. 'tau'), needed for reconstruction formula
    qpspecgram : the Spectrogram (i.e. energies) associated to this Qp-transform; time pixels are interpolated to 'times'
    filseries : if 'filmethod' is not None this is a TimeSeries object with the filtered data
	"""	
    def __init__(self, data, times, outseg=None, whiten=True, 
        derivative=False, 
        frange=DEFAULT_FRANGE, 
        alpha=DEFAULT_ALPHA, alpha_find_Qp=DEFAULT_ALPHA, 
        energy_density_threshold=DEFAULT_ENERGY_DENSITY_THRESHOLD, Qp_finder='energy_density',
        qrange=DEFAULT_QRANGE, prange=DEFAULT_PRANGE, max_Qp_iters=DEFAULT_MAX_Qp_ITERS, 
        filmethod='highpass_threshold_filtering', denoising_threshold=DEFAULT_ENERGY_DENSITY_THRESHOLD):
        
        self.series = TimeSeries(data=numpy.asarray(data), times=numpy.asarray(times))
    
        # whiten with full data segment, and then apply outseg cut 
        if whiten is True:
            self.series = self.series.whiten() 

        if outseg is not None:
            self.series = self.series.crop(outseg[0], outseg[-1])

        self.times=self.series.times.value

        self.qpgram=qp_scan(self.series, self.times, frange=frange, alpha=alpha, alpha_find_Qp=alpha_find_Qp, energy_density_threshold=energy_density_threshold, to_return=Qp_finder,  qrange=qrange, prange=prange, max_Qp_iters=max_Qp_iters)

        self.Q=self.qpgram.Q

        self.p=self.qpgram.p

        self.peak=self.qpgram.peak

        self.energy_density=self.qpgram.energy_density

        self.TF_area=self.qpgram.TF_area
            
        if(derivative==True): 
            fft_series=scipy.fft.fft(self.series.value)
            fft_freqs=scipy.fft.fftfreq(len(fft_series)) * self.series.sample_rate.value
            self.qpgram_derivative=QpPlane(Q=self.Q, p=self.p, frange=frange, duration=self.series.duration.value, sampling=self.series.sample_rate.value, alpha=alpha).transform(fft_series, fft_freqs, self.series.times.value[0], energy_density_threshold, derivative=derivative)

        self.qpspecgram=qpgram_to_qpspecgram(qp_scan(self.series, self.times, frange=frange, alpha=alpha, alpha_find_Qp=alpha_find_Qp, energy_density_threshold=energy_density_threshold, to_return=Qp_finder,  qrange=[self.Q,self.Q], prange=[self.p,self.p], max_Qp_iters=max_Qp_iters, original_sampling_rate=True), to_return='energies')
        


        if(filmethod is not None):
            self.filseries=iQpt_filtering(self.series, self.times, self.qpspecgram, filmethod, denoising_threshold)





# functions

def find_Q(Q, *args):
    """
    Function to find Q maximising the energy density ot the energy peak
    """
    p, frange, duration, sampling, alpha, energy_density_threshold, fft_series, fft_freqs, start_time, to_return=args
    qpgram=QpPlane(Q=Q, p=p, frange=frange, duration=duration, sampling=sampling, alpha=alpha).transform(fft_series, fft_freqs, start_time, energy_density_threshold)
    if(to_return=='energy_density'):
        return -qpgram.energy_density
    elif(to_return=='energy_peak'):
        return -qpgram.peak['energy']
    elif(to_return=='weighted_energy_density'):
        return -qpgram.weighted_energy_density


def find_p(p, *args):
    """
    Function to find p maximising the energy density ot the energy peak
    """
    Q, frange, duration, sampling, alpha, energy_density_threshold, fft_series, fft_freqs, start_time, to_return=args
    qpgram=QpPlane(Q=Q, p=p, frange=frange, duration=duration, sampling=sampling, alpha=alpha).transform(fft_series, fft_freqs, start_time, energy_density_threshold)
    if(to_return=='energy_density'):
        return -qpgram.energy_density
    elif(to_return=='energy_peak'):
        return -qpgram.peak['energy']
    elif(to_return=='weighted_energy_density'):
        return -qpgram.weighted_energy_density

    

def find_Qp(Qp, *args):
    """
    Function to find p maximising the energy density ot the energy peak
    """
    Q=Qp[0]
    p=Qp[-1]
    frange, duration, sampling, alpha, energy_density_threshold, fft_series, fft_freqs, start_time, to_return=args
    qpgram=QpPlane(Q=Q, p=p, frange=frange, duration=duration, sampling=sampling, alpha=alpha).transform(fft_series, fft_freqs, start_time, energy_density_threshold)
    if(to_return=='energy_density'):
        return -qpgram.energy_density
    elif(to_return=='energy_peak'):
        return -qpgram.peak['energy']
    elif(to_return=='weighted_energy_density'):
        return -qpgram.weighted_energy_density


def qp_scan(series, times, frange=DEFAULT_FRANGE, alpha=DEFAULT_ALPHA, alpha_find_Qp=DEFAULT_ALPHA, energy_density_threshold=DEFAULT_ENERGY_DENSITY_THRESHOLD, to_return='energy_density',
    qrange=DEFAULT_QRANGE, prange=DEFAULT_PRANGE, max_Qp_iters=DEFAULT_MAX_Qp_ITERS, max_basinhopping_iters=DEFAULT_MAX_BASINHOPPING_ITERS, original_sampling_rate=False):
    """
    This function find the values of Q and p which maximize the energy density (or the energy of the loudest tile, if the energy is below 'energy_density_threshold')
    Then it returns the qpgram obtained with these Q and p
    Parameters
    ----------
    series : the TimeSeries object with the data needed for Qp-transform
    times : the time array corresponding to 'series'
    frange : the frequency range on which the Qp-transform is calculated
    alpha : the alpha used for calculate the final Qp-transform (i.e. once Q and p have been found)
    alpha_find_Qp : the alpha used for finding the Q and p values throug the maximization of the qpgram energy density;
        this is usually higher than 'alpha'
        IMPORTANT: number of time-frequency points (and so computational time) scales as 1/(alpha**2)
    energy_density_threshold: the threshold above which the energy density of the time-frequency map is calculated
    qrange : the selected range for finding the Q value which maximise the energy density
    prange : the selected range for finding the p value which maximise the energy density
    max_Qp_iters : max number of maximizer iterations for finding (Q,p)
    max_basinhopping_iters : max number of 'niter' given to basinhopping algorithm (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
    original_sampling_rate: if 'True' the transform is calculated at each frequency for a number of time points equal to that of the starting time-series
    Returns
    -------
    QpGram object with the found Q and p
    """
    duration=times[-1]-times[0]
    sampling=numpy.ceil(1./(times[1]-times[0]))
    fft_series=scipy.fft.fft(series)
    fft_freqs=scipy.fft.fftfreq(len(series)) * sampling
       
    if(qrange[0]==qrange[-1] and prange[0]==prange[-1]):
        Q=qrange[0]
        p=prange[0]
    
    else:
        # find Q and p together
        if(prange==[0.,0.]):
            p=0.
            #to_return='energy_density'
            Q_bnds=scipy.optimize.Bounds(lb=qrange[0], ub=qrange[-1]) 
            minimizer_kwargs = {'method':DEFAULT_MINIMIZE_METHOD, 'args':(0., frange, duration, sampling, alpha_find_Qp, energy_density_threshold, fft_series, fft_freqs, times[0], to_return), 'bounds':Q_bnds, 'options':{"maxiter":max_Qp_iters}}
            res_Q=basinhopping(find_Q, x0=numpy.sqrt(qrange[-1]*qrange[0]), minimizer_kwargs=minimizer_kwargs, niter=max_basinhopping_iters)
            if(res_Q.fun==0. or res_Q.success==False):  
                warnings.warn("'energy_density' does not converge for finding (Q,p), 'energy_peak' is used instead")
                to_return='energy_peak'
                Q_bnds=scipy.optimize.Bounds(lb=qrange[0], ub=qrange[-1]) 
                minimizer_kwargs = {'method':DEFAULT_MINIMIZE_METHOD, 'args':(0., frange, duration, sampling, alpha_find_Qp, energy_density_threshold, fft_series, fft_freqs, times[0], to_return), 'bounds':Q_bnds, 'options':{"maxiter":max_Qp_iters}}
                res_Q=basinhopping(find_Q, x0=numpy.sqrt(qrange[-1]*qrange[0]), minimizer_kwargs=minimizer_kwargs, niter=max_basinhopping_iters)
            Q=res_Q.x[0]

        elif(prange!=[0.,0.]):
            if(prange is not None):
                Qp_bnds=scipy.optimize.Bounds(lb=[qrange[0],prange[0]], ub=[qrange[-1],prange[-1]]) 
            elif(prange is None):
                Qp_bnds=scipy.optimize.Bounds(lb=[qrange[0],0], ub=[qrange[-1],1./qrange[0]]) # pensaci
            #to_return='energy_density'
            
            minimizer_kwargs = {'method':DEFAULT_MINIMIZE_METHOD, 'args':(frange, duration, sampling, alpha_find_Qp, energy_density_threshold, fft_series, fft_freqs, times[0], to_return), 'bounds':Qp_bnds, 'options':{"maxiter":max_Qp_iters}}
            res_Qp=basinhopping(find_Qp, x0=[numpy.sqrt(qrange[-1]*qrange[0]), 0.], minimizer_kwargs=minimizer_kwargs, niter=max_basinhopping_iters)
            
            if(res_Qp.fun==0. or res_Qp.success==False): 
                warnings.warn("'energy_density' does not converge for finding (Q,p), 'energy_peak' is used instead")
                to_return='energy_peak' 
                minimizer_kwargs = {'method':DEFAULT_MINIMIZE_METHOD, 'args':(frange, duration, sampling, alpha_find_Qp, energy_density_threshold, fft_series, fft_freqs, times[0], to_return), 'bounds':Qp_bnds, 'options':{"maxiter":max_Qp_iters}}
                res_Qp=basinhopping(find_Qp, x0=[numpy.sqrt(qrange[-1]*qrange[0]), 0.], minimizer_kwargs=minimizer_kwargs, niter=max_basinhopping_iters)
            Q=res_Qp.x[0]
            p=res_Qp.x[1]

    qpgram=QpPlane(Q=Q, p=p, frange=frange, duration=duration, sampling=sampling, alpha=alpha).transform(fft_series, fft_freqs, times[0],energy_density_threshold, original_sampling_rate=original_sampling_rate)
    
    return qpgram

def qpgram_to_qpspecgram(qpgram,to_return):

    frequencies = qpgram.frequencies
    
    if(to_return=='energies'): # energies for Spectrogram
        qpobject=qpgram.energies
    
    elif(to_return=='plane'): # plane for specific purposes (e.g. interpolate likelihood)
        qpobject=qpgram.plane

    else:
        warnings.warn("'to_return' has to be 'plane' or 'energies', None is returned")
        return None
    
    dtype = qpobject[0].dtype
    nx = len(qpobject[0].value)
    ny = frequencies.size
    qpmatrix = Spectrogram(numpy.empty((nx, ny), dtype=dtype), times=qpobject[0].times, frequencies=frequencies)
    qpmatrix.Q=qpgram.Q
    qpmatrix.p=qpgram.p

    for i in range(ny):
        qpmatrix[:, i] = qpobject[i].value
    
    return qpmatrix	    


def qpobject_filtering(reference_qpobject, application_qpobject, filmethod, denoising_threshold):
    """
    This function takes the 'application_qpobject' and sets to zero the pixels corresponding to that of 
        a given 'reference_qpobject' above/below some given threshold(s)  
    Notice that it's quite common to set to zero pixels from a given qpobject (i.e. 'application_qpobject')
        basing on the values of the pixels of another qpobject (i.e. 'reference_qpobject')
        For istance, 'application_qpobject' could be a QpGram.plane which pixels are selected according to the energies
        of the corresponding QpGram.energies which is given as 'reference_qpobject'
    Parameters
    ----------
    reference_qpobject : a QpGram or Spectrogram object taken as reference for finding which pixels have to be set to zero
    application_qpobject : a QpGram or Spectrogram object to be filtered
    filmethod : method to filter the 'application_qpobject'
        Filtering methods can be
            'highpass_threshold_filtering'->takes all time-frequency regions with energy above 'denoising_threshold'
            'lowpass_threshold_filtering'->takes all time-frequency regions with energy below 'denoising_threshold'
            'rangepass_threshold_filtering'->in that case 'denoising_threshold' should be made by two elements, 
            filmethod takes all time-frequency regions with energy between 'denoising_threshold[0]' and 'denoising_threshold[-1]'
    denoising_threshold : the thresold taken as reference for 'filmethod'
        if 'filmethod'=='rangepass_threshold_filtering' then 'denoising_threshold' should be made by two elements
    Returns
    ----------
    filqpobject : the filtered 'application_qpobject'
    """  
    if(filmethod=='highpass_threshold_filtering'):
        boolean_qpobject = reference_qpobject<denoising_threshold
        
    elif(filmethod=='lowpass_threshold_filtering'):
        boolean_qpobject = reference_qpobject>denoising_threshold
        
    elif(filmethod=='rangepass_threshold_filtering'):
        if(len(denoising_threshold)!=2):
            warnings.warn("For range pass threshold filtering threshold should have two elements, original 'application_qpobject' is returned")
            return application_qpobject
        hptf_boolean = reference_qpobject<denoising_threshold[0]
        lptf_boolean = reference_qpobject>denoising_threshold[1]
        boolean_qpobject = numpy.logical_or(hptf_boolean, lptf_boolean)
    
    else:
        warnings.warn("'filmethod' should be 'highpass_threshold_filtering', 'lowpass_threshold_filtering' or 'rangepass_threshold_filtering', original 'application_qpobject' is returned")
        return application_qpobject 

    filqpobject=application_qpobject.copy()

    filqpobject.Q=application_qpobject.Q
    filqpobject.p=application_qpobject.p
    filqpobject.value[boolean_qpobject]=0.
    
    return filqpobject

	
def filfreqboundaries(filqpobject): 
    """
    This function takes a filtered qpbject 'filqpobject' and finds at each time  
    the frequencies delimiting non-zero regions
    Notice that 'filqpobject' should be the result of 'qpgram_time_interpolate' function
    Parameters
    ----------
    filqpobject : filtered qpbject with zero and non-zero pixels
    Returns
    freqboundaries_lst : a list containing a sublist for each time of the 'filqpobject';
        these sublists contains the time as first element, then (eventually) the frequency boundaries
        of non-zero pixels at that time, in increasing order
        Each sublist has an odd number of elements (time + an even number of frequencies)
    ----------
    """  	
    freqboundaries_lst=[]
    times=filqpobject.times.value
    frequencies=filqpobject.frequencies.value


    for i in range(len(times)):
        i_lst=[times[i]]
        if(numpy.linalg.norm(filqpobject[i, :])==0.):	
            freqboundaries_lst.append(i_lst)
            continue

        nonzero=numpy.array(numpy.where(filqpobject[i, :]!=0)[0], dtype=int)

        i_lst.append(frequencies[nonzero[0]])
        if(nonzero[-1]==nonzero[0]+len(nonzero)-1): # uniqpue interval simply connected
            i_lst.append(frequencies[nonzero[-1]])
        else:
            for l in range(1, len(nonzero)):
                if(nonzero[l]-nonzero[l-1]>1):
                    i_lst.append(frequencies[nonzero[l-1]])
                    i_lst.append(frequencies[nonzero[l]])
            i_lst.append(frequencies[nonzero[-1]])

        freqboundaries_lst.append(i_lst)
                                    
    return freqboundaries_lst


def iQpt_filtering(series, times, qpobject, filmethod, denoising_threshold):
    """
    This function takes as input a set of data ('series') and filters it 
        according to the filtering of the corresponding 'qpobject' 
    Parameters
    ----------
    series : the data to be filtered
    times : the corresponding time array; notice that this should be the time array of the qpobject as well
    qpobject : the qpobject to be filtered; according to 'filmethod' and 'denoising_threshold'
        this gives the time-frequency regions to be set to zero
    filmethod : method to filter the 'qpobject' and then the data accordingly
        Filtering methods can be
            'highpass_threshold_filtering'->takes all time-frequency regions with energy above 'denoising_threshold'
            'lowpass_threshold_filtering'->takes all time-frequency regions with energy below 'denoising_threshold'
            'rangepass_threshold_filtering'->in that case 'denoising_threshold' should be made by two elements, 
            filmethod takes all time-frequency regions with energy between 'denoising_threshold[0]' and 'denoising_threshold[-1]'
    denoising_threshold : the thresold taken as reference for 'filmethod'
        if 'filmethod'=='rangepass_threshold_filtering' then 'denoising_threshold' should be made by two elements
    Returns
    iqptfilseries : the filtered data
    ----------
    """  	
   
    f_s=numpy.ceil(1./(times[1]-times[0])) # Hz
    frequencies=scipy.fft.fftfreq(len(times))*f_s
    
    fft_series=scipy.fft.fft(series)

    filqpobject=qpobject_filtering(reference_qpobject=qpobject, application_qpobject=qpobject, filmethod=filmethod, denoising_threshold=denoising_threshold)
    freqboundaries=filfreqboundaries(filqpobject)

    Qtilde=qpobject.Q/(numpy.sqrt(1+1.j*2.*qpobject.p*qpobject.Q))

    fil_series=numpy.zeros(len(times))
    for i in range(len(fil_series)):	
        if(len(freqboundaries[i])>1):
            window=numpy.zeros(len(frequencies), dtype=complex)
            for k in range(1, len(freqboundaries[i])):
                window+=((-1.)**(k-1))*special.erf((Qtilde/2.)*((frequencies/freqboundaries[i][k])-1.)) 
            window=window/numpy.real(special.erf(Qtilde/2.)) 
            fft_series_i=fft_series*window
            fil_series[i]=numpy.real(scipy.fft.ifft(fft_series_i)[i])

    iqptfilseries=TimeSeries(data=fil_series, times=times)

    return iqptfilseries


def inverse_Qp_transform(qpmatrix_derivatives, times): 
    """
    This function takes the time-derivative of a QpGram interpolated to 'times'
    and returns a corresponding TimeSeries object obtained with pseudo-inverse Qp-transform
    Parameters
    ----------
    qpmatrix_derivatives : the time-derivative of a QpGram interpolated to 'times'
    times : the time array associated to 'qpmatrix_derivatives'
    Returns
    ----------
    iqptseries : the TimeSeries object obtained with pseudo-inverse Qp-transform
    """  	
    frequencies=qpmatrix_derivatives.frequencies

    Qtilde=qpmatrix_derivatives.Q/(numpy.sqrt(1+1.j*2.*qpmatrix_derivatives.p*qpmatrix_derivatives.Q))
    
    norm=numpy.sqrt(times[1]-times[0])/(((2./((qpmatrix_derivatives.Q**2)*numpy.pi))**(1./4.))*numpy.real(special.erf(Qtilde/2.)))

    iqpt_value=numpy.zeros(len(times), dtype=float)

    for i in range(len(times)):
        iqpt_value[i]=numpy.real(integrate.simpson(qpmatrix_derivatives[i,:].value/(1.j*((2.*numpy.pi*frequencies)**(3./2.))) , frequencies))
        
    iqpt_value=2.*norm*iqpt_value

    iqptseries=TimeSeries(data=iqpt_value, times=times)

    return iqptseries











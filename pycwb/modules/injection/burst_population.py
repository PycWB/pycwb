"""LF burst waveforms with cWB source-amplitude and polarization conventions.

The random-number stream is NumPy's, not ROOT's. WNB frequency is the lower
band edge; duration is the Gaussian amplitude-envelope standard deviation.
"""
import numpy as np
from pycwb.types.time_series import TimeSeries


def _wnb(frequency, bandwidth, duration, seed, sample_rate, n):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(size=n))
    df = sample_rate / n
    center = int(frequency / df) + int(bandwidth / (2 * df))
    half = int(bandwidth / (2 * df))
    if half < 1 or center + half >= len(spectrum):
        raise ValueError('WNB band must fit inside the sampled frequency range')
    shifted = np.zeros_like(spectrum)
    # cWB's packed real FFT stores the Nyquist real component beside DC;
    # GetWNB copies both to the heterodyne center, including the imaginary part.
    shifted[center] = spectrum[0].real + 1j*spectrum[-1].real
    for k in range(1, half):
        shifted[center+k] = spectrum[k]
        shifted[center-k] = spectrum[k].conjugate()
    signal = np.fft.irfft(shifted, n=n)
    t = (np.arange(n)+1) / sample_rate - 0.5
    signal *= np.exp(-0.5 * (t / duration)**2)
    return signal / np.sqrt(2 * np.sum(signal**2) / sample_rate)


def get_td_waveform(**p):
    """Return polarizations; hrss is defined BEFORE SGE inclination factors."""
    rate = 1.0 / float(p.get('delta_t', 1/16384))
    n = int(round(rate))
    t = (np.arange(n) - n//2) / rate
    kind = p['approximant']
    if kind == 'WNB':
        hp = _wnb(p['frequency'], p['bandwidth'], p['duration'], p['pseed'], rate, n)
        hc = _wnb(p['frequency'], p['bandwidth'], p['duration'], p['xseed'], rate, n)
    elif kind in ('SG', 'SGE'):
        sigma = p['Q'] / (2*np.pi*p['frequency'])
        m = min(int(6*sigma*rate), n//2-2)
        envelope = np.exp(-0.5*(t/sigma)**2) * (np.abs(np.arange(n)-n//2) < m)
        hp = envelope * np.sin(2*np.pi*p['frequency']*t)
        hc = envelope * np.cos(2*np.pi*p['frequency']*t)
        # cWB AddSG/CG normalize using the positive half, including t=0.
        pos = np.arange(m)/rate
        for signal, trig in ((hp,np.sin),(hc,np.cos)):
            s = 2*np.exp(-0.5*(pos/sigma)**2)*trig(2*np.pi*p['frequency']*pos)
            signal *= np.sqrt(rate / np.sum(s*s))
        if kind == 'SG':
            hc[:] = 0
    elif kind == 'GA':
        hp = np.exp(-(((np.arange(n)+1)/rate-0.5)/p['duration'])**2)
        hc = np.zeros(n)
    else:
        raise ValueError(f'Unsupported LF burst family: {kind}')
    amplitude = float(p.get('hrss', 5e-23)) / np.sqrt(np.sum(hp*hp+hc*hc)/rate)
    hp *= amplitude
    hc *= amplitude
    if kind == 'SGE':
        cosine = np.cos(p['iota'])
        hp *= (1+cosine*cosine)/2
        hc *= cosine
    return {'type':'polarizations', 'hp':TimeSeries(hp,dt=1/rate,t0=-0.5),
            'hc':TimeSeries(hc,dt=1/rate,t0=-0.5)}

"""Scale burst injections to a target network SNR using clean-data WDM noise."""
from copy import copy

import numpy as np
from pycwb.types.time_series import TimeSeries
from .snr_energy import cwb_snr_energy
from pycwb.modules.injection.strain import generate_strain_from_injection
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.data_conditioning.whitening import whitening_python
from pycwb.modules.data_conditioning.injection_whitening import whiten_injection_strain


def _window_energy(series, gps, half_window):
    rate=float(series.sample_rate); origin=float(series.t0); values=series.data
    lo=max(0,int((gps-origin-half_window)*rate)); hi=min(len(values),int((gps-origin+half_window)*rate))
    energy=values[lo:hi]**2
    if energy.sum()<=0:
        return 0.
    center=origin+float(np.dot(np.arange(lo,hi),energy)/energy.sum())/rate
    lo=max(0,int((center-origin-half_window)*rate)); hi=min(len(values),int((center-origin+half_window)*rate))
    return float(np.sum(values[lo:hi]**2))


def target_snr_scales(config, segment, clean_data):
    """Return per-injection scales; do not contaminate the noise estimate.

    Match cWB simulation=5: estimate WDM noise before regression/injection,
    whiten the signal, and sum detector energies inside iwindow/2. Nearby
    injections are rejected because a shared window cannot define their SNRs.
    """
    injections=segment.injections
    targets=[float(p.get('target_snr',p.get('targeted_snr',0))) for p in injections]
    if not any(targets):
        return np.ones(len(injections))
    if any(t<0 or not np.isfinite(t) for t in targets):
        raise ValueError('Target SNR must be finite and nonnegative')
    half=float(config.iwindow)/2
    times=np.sort([p['gps_time'] for p in injections])
    if len(times)>1 and np.min(np.diff(times))<=2*half:
        raise ValueError('Target-SNR injections must have nonoverlapping SNR windows')
    buffers=[TimeSeries(np.zeros(len(d.data)),dt=d.dt,t0=d.t0) for d in clean_data]
    for p in injections:
        signals=generate_strain_from_injection(p,config,segment.sample_rate,segment.ifos)
        for buffer,signal in zip(buffers,signals):
            buffer.inject(signal,copy=False)
    reference_mode = getattr(config, "injection_resampling", "fft") == "cwb"
    signal_config = config
    if reference_mode:
        # cWB calibrates noise only; its separately added MDC is not calibrated.
        signal_config = copy(config)
        signal_config.dcCal = [1.0] * len(clean_data)
    energy=np.zeros(len(injections))
    for i,(data,signal) in enumerate(zip(clean_data,buffers)):
        noise=check_and_resample_py(data.copy(),config,i)
        _,rms=whitening_python(config,noise,apply_bandpass=not reference_mode)
        signal=check_and_resample_py(signal,signal_config,i)
        white,_=whiten_injection_strain(config,signal,rms)
        if reference_mode:
            energy += [cwb_snr_energy(white,p['gps_time'],half,config.fLow,config.fHigh) for p in injections]
        else:
            energy += [_window_energy(white,p['gps_time'],half) for p in injections]
    scales=np.ones(len(injections))
    for i,(p,target) in enumerate(zip(injections,targets)):
        if target:
            if energy[i]<=0 or not np.isfinite(energy[i]):
                raise ValueError('Cannot scale an injection with zero/nonfinite network SNR')
            scales[i]=target/np.sqrt(energy[i])
            p['snr_scale']=float(scales[i])
            p['unscaled_network_snr']=float(np.sqrt(energy[i]))
    return scales

"""Scientific convention checks for LF burst populations."""
import numpy as np
import pytest
from pycwb.modules.injection.burst_population import get_td_waveform
from pycwb.modules.injection.snr_population import _window_energy
from pycwb.types.time_series import TimeSeries


def energy(wave):
    return np.sum(wave.data**2)*wave.dt


@pytest.mark.parametrize('kind,parameters',[
    ('GA',dict(duration=.001)),('SG',dict(frequency=100,Q=9)),
    ('SGE',dict(frequency=100,Q=9,iota=0)),
    ('WNB',dict(frequency=150,bandwidth=100,duration=.1,pseed=1,xseed=2)),
])
def test_source_amplitude(kind,parameters):
    w=get_td_waveform(approximant=kind,hrss=2e-22,**parameters)
    assert np.sqrt(energy(w['hp'])+energy(w['hc']))==pytest.approx(2e-22)
    if kind=='WNB':
        assert energy(w['hp'])==pytest.approx(energy(w['hc']),rel=1e-12,abs=0)
    if kind in ('GA','SG'):
        assert np.count_nonzero(w['hc'].data)==0


def test_sge_inclination_does_not_renormalize_source_amplitude():
    a=get_td_waveform(approximant='SGE',frequency=70,Q=3,iota=0,hrss=1e-22)
    b=get_td_waveform(approximant='SGE',frequency=70,Q=3,iota=np.pi/2,hrss=1e-22)
    np.testing.assert_allclose(b['hp'].data,a['hp'].data/2,atol=0)
    assert energy(b['hc'])<energy(a['hc'])*1e-28


def test_wnb_duration_is_amplitude_standard_deviation():
    # Wide band averages stochastic power fluctuations; energy envelope has sigma/sqrt(2).
    moments=[]
    for seed in range(32):
        w=get_td_waveform(approximant='WNB',frequency=300,bandwidth=800,duration=.06,pseed=seed,xseed=seed+50)['hp']
        t=(np.arange(len(w.data))+1)*w.dt-.5
        moments.append(np.sum(t*t*w.data*w.data)/np.sum(w.data*w.data))
    assert np.sqrt(np.mean(moments))==pytest.approx(.06/np.sqrt(2),rel=.06)


def test_snr_energy_is_sum_of_samples_not_time_integral():
    w=TimeSeries(np.array([0.,3.,4.,0.]),dt=.25,t0=100.)
    assert _window_energy(w,100.5,1.)==25.


def test_injection_whitening_uses_data_phase_and_noise_anchor_conventions():
    from types import SimpleNamespace
    from pycwb.modules.data_conditioning.whitening import whiten_wavelet
    from pycwb.modules.data_conditioning.injection_whitening import whiten_injection_strain
    config=SimpleNamespace(l_white=5,l_high=5,WDM_beta_order=6,WDM_precision=10,
                           whiteWindow=8.,whiteStride=4.,segEdge=2.,fLow=0.,fHigh=512.)
    rng=np.random.default_rng(92)
    samples=rng.normal(size=40960)*np.linspace(1,4,40960)
    signal=TimeSeries(samples,dt=1/1024,t0=1387221730.)
    expected,rms=whiten_wavelet(config,signal)
    actual,_=whiten_injection_strain(config,signal,rms)
    np.testing.assert_allclose(actual.data,expected.data,rtol=1e-11,atol=1e-11)


def test_single_zero_lag_and_invalid_zero_step():
    from pycwb.types.job import WaveSegment
    segment=WaveSegment(index=1,ifos=['L1','H1'],analyze_start=100,analyze_end=200,sample_rate=16384,
                        seg_edge=10,lag_size=1,lag_step=0,lag_off=0,lag_max=0)
    np.testing.assert_array_equal(segment.lag_shifts,[[0.,0.]])
    segment.lag_size=2
    with pytest.raises(ValueError,match='lag_step'):
        segment._compute_lag_shifts()


def test_exact_length_segment_has_positive_unique_job_id():
    from pycwb.modules.job_segment.dq_segment import get_job_list
    jobs=get_job_list(['L1','H1'],([100,300],[220,420]),100,50,seg_edge=10,sample_rate=16384,index_start=4)
    assert [j.index for j in jobs]==[5,6]


def test_target_network_snr_scales_actual_whitened_detector_signals():
    from types import SimpleNamespace
    from pycwb.modules.injection.snr_population import target_snr_scales
    from pycwb.modules.injection.strain import generate_strain_from_injection
    from pycwb.modules.read_data.data_check import check_and_resample_py
    from pycwb.modules.data_conditioning.whitening import whiten_wavelet
    from pycwb.modules.data_conditioning.injection_whitening import whiten_injection_strain
    config=SimpleNamespace(inRate=1024,fResample=0,levelR=0,dcCal=[1,1],
        l_white=5,l_high=5,WDM_beta_order=6,WDM_precision=10,whiteWindow=8.,whiteStride=4.,
        segEdge=2.,fLow=16.,fHigh=512.,iwindow=5.,detector_geometry='cwb_6.4.6.9',
        injection={'generator':'pycwb.modules.injection.burst_population.get_td_waveform'})
    p=dict(approximant='SGE',frequency=100,Q=9,iota=.7,hrss=1e-22,
           ra=1.,dec=.3,pol=.8,gps_time=1387221750.,target_snr=20.)
    from pycwb.types.detector import Detector
    detectors = {ifo: Detector(ifo, geometry_model=f"{ifo}:cwb") for ifo in ['L1', 'H1']}
    config.get_detectors = lambda names: tuple(detectors[name] for name in names)
    segment=SimpleNamespace(injections=[p],sample_rate=1024,ifos=['L1','H1'])
    rng=np.random.default_rng(7)
    data=[TimeSeries(rng.normal(size=40960)*1e-22,dt=1/1024,t0=1387221730.) for _ in range(2)]
    original=[d.data.copy() for d in data]
    scale=target_snr_scales(config,segment,data)[0]
    signals=generate_strain_from_injection(p,config,1024,segment.ifos)
    measured=0.
    for i,(noise,signal) in enumerate(zip(data,signals)):
        np.testing.assert_array_equal(noise.data,original[i])
        signal.data*=scale
        full=TimeSeries(np.zeros(40960),dt=1/1024,t0=1387221730.).inject(signal)
        _,rms=whiten_wavelet(config,check_and_resample_py(noise.copy(),config,i))
        white,_=whiten_injection_strain(config,check_and_resample_py(full,config,i),rms)
        measured+=_window_energy(white,p['gps_time'],2.5)
    assert np.sqrt(measured)==pytest.approx(20.,rel=1e-10)


def test_gps_grid_rate_does_not_stretch_or_clip_at_edges():
    from pycwb.modules.injection.injection import distribute_inj_on_gps_grid
    intervals=[dict(start=101.,end=361.,duration=260.,job_id=1,shift=[0.,0.])]
    rows,trials=distribute_inj_on_gps_grid([{} for _ in range(5)],1/50.,0.,intervals)
    assert trials==1
    assert [r['gps_time'] for r in rows]==[150.,200.,250.,300.,350.]
    np.random.seed(17)
    rows,trials=distribute_inj_on_gps_grid([{} for _ in range(10000)],1/50.,10.,intervals)
    times=np.array([r['gps_time'] for r in rows])
    assert np.all((times>=101)&(times<361))
    assert not np.any((times==101)|(times==361))
    residual=(times+25)%50-25
    assert np.max(abs(residual))<=10.


def test_gps_grid_does_not_redraw_same_anchor_across_cat2_gaps():
    from pycwb.modules.injection.injection import distribute_inj_on_gps_grid
    keep=[dict(start=a,end=b,duration=b-a,job_id=1,shift=[0.,0.]) for a,b in [(101.,149.),(151.,199.)]]
    np.random.seed(27)
    rows,_=distribute_inj_on_gps_grid([{} for _ in range(1000)],1/50.,10.,keep)
    keys=[(r['trial_idx'],round(r['gps_time']/50)) for r in rows]
    assert len(keys)==len(set(keys))

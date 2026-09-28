from types import SimpleNamespace
from dataclasses import replace
import json
import numpy as np
import pytest
import jsonschema

from pycwb.types.time_series import TimeSeries
from pycwb.types.noise_rms import NoiseRMSMap
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.modules.data_conditioning.noise import lookup_pixel_noise_rms
from pycwb.modules.conditioning_plugins.api import run_hooks, subtract_intervals, save_diagnostics
from pycwb.modules.conditioning_plugins.cwb_gating import gate_intervals
from pycwb.modules.conditioning_plugins.o3a_conditioning import correct_strain


def noise():
    return NoiseRMSMap(np.ones((129, 8))*2, 1., 1/64., 1000., 64*128, {}, 1000., .125, 1000.)


def test_no_hooks_preserves_inputs():
    strain = TimeSeries(data=np.arange(100.), t0=1000., dt=.01)
    rms = noise()
    result = run_hooks(SimpleNamespace(), SimpleNamespace(ifos=['L1']), [strain], [rms])
    assert result.strains[0] is strain and result.noise_rms[0] is rms
    assert not result.diagnostics and not result.excluded_intervals


def test_gate_excludes_pixels_without_changing_strain(tmp_path):
    x = np.zeros(64*128)
    x[30*128:30*128+16] = 2000
    strain = TimeSeries(data=x.copy(), t0=1000., dt=1/128.)
    config = SimpleNamespace(segEdge=10, selection={'time_vetoes':[
        {'module':'pycwb.modules.conditioning_plugins.cwb_gating'}]})
    result = run_hooks(config, SimpleNamespace(ifos=['L1']), [strain], [noise()])
    assert result.excluded_intervals == [(1028., 1032.)]
    np.testing.assert_array_equal(result.strains[0].data, x)
    assert subtract_intervals(None, result.excluded_intervals, 1010, 1054) == [(1010,1028),(1032,1054)]
    assert subtract_intervals([], result.excluded_intervals, 1010,1054) == []
    save_diagnostics(result,tmp_path)
    assert json.loads((tmp_path/'diagnostics.json').read_text())['excluded_intervals']==[[1028,1032]]


def test_gate_edges_and_no_activity():
    x=np.zeros(64*128);x[128:256]=1e4
    assert gate_intervals(TimeSeries(data=x,t0=1000.,dt=1/128.),10)==[]
    assert gate_intervals(TimeSeries(data=x*0,t0=1000.,dt=1/128.),10)==[]


def test_plugin_options_fail_closed():
    config=SimpleNamespace(selection={'time_vetoes':[
        {'module':'pycwb.modules.conditioning_plugins.cwb_gating','options':{'typo':1}}]})
    with pytest.raises(jsonschema.ValidationError):
        run_hooks(config,SimpleNamespace(ifos=[]),[],[])
    config=SimpleNamespace(conditioning={'post_whitening':[
        {'module':'pycwb.modules.conditioning_plugins.cwb_gating'}]})
    with pytest.raises(ValueError,match='does not implement'):
        run_hooks(config,SimpleNamespace(ifos=[]),[],[])


def test_noise_variation_overlap_and_lag_index():
    rms=noise()
    # 32 Hz-wide pixels: [16,48], [48,80], [0,64] at different resolutions.
    var=TimeFrequencyMap(data=np.full((1,64*64),.5,dtype=np.float32),
                         is_whitened=False,dt=1/64.,df=32.,start=1000.,stop=1064.,
                         f_low=16.,f_high=48.,edge=10.,wavelet=None)
    corrected=replace(rms,variation=var)
    frequencies=np.array([1,2,1]);rates=np.array([64.,64.,128.]);layers=np.array([3,3,3])
    idx=(rates*layers*20).astype(int)[:,None]
    baseline=lookup_pixel_noise_rms(frequencies,idx,layers,rates,[rms])[:,0]
    out=lookup_pixel_noise_rms(frequencies,idx,layers,rates,[corrected])[:,0]
    # Last pixel is [32,96]: quarter overlaps [16,48].
    np.testing.assert_allclose(out/baseline,[2,1,1/np.sqrt(.75+.25*.25)])
    var.data[:,30*64:]=1
    out_later=lookup_pixel_noise_rms(frequencies,idx+(rates*layers*15).astype(int)[:,None],layers,rates,[corrected])[:,0]
    np.testing.assert_array_equal(out_later,baseline)


def test_zero_signal_is_finite_identity():
    x=TimeSeries(data=np.zeros(64*128),t0=1000.,dt=1/128.)
    corrected,var=correct_strain(x,10)
    np.testing.assert_array_equal(corrected.data,x.data)
    np.testing.assert_array_equal(var.data,1.)
    assert type(var) is TimeFrequencyMap
    assert var.data.shape == (1,64*64) and var.data.dtype == np.float32
    assert (var.start,var.stop,var.dt,var.f_low,var.f_high) == (1000.,1064.,1/64.,16.,48.)


def test_variation_changes_only_selected_detector(tmp_path):
    x=TimeSeries(data=np.random.default_rng(21).normal(size=64*128),t0=1000.,dt=1/128.)
    rms=noise()
    config=SimpleNamespace(segEdge=10,conditioning={'post_whitening':[
        {'module':'pycwb.modules.conditioning_plugins.o3a_conditioning','options':{'detectors':['L1']}}]})
    result=run_hooks(config,SimpleNamespace(ifos=['L1','H1']),[x,x],[rms,rms])
    assert result.strains[1] is x and result.noise_rms[1] is rms
    assert result.noise_rms[0].variation is not None and rms.variation is None
    assert np.isfinite(result.strains[0].data).all()
    save_diagnostics(result,tmp_path)
    metadata=json.loads((tmp_path/'diagnostics.json').read_text())
    assert metadata['noise_variation'] == [dict(start=1000.,rate=64.,low=16.,high=48.),None]
    with np.load(tmp_path/'noise_variation.npz') as saved:
        assert saved['nvar_0'].shape == (64*64,)
        assert saved['nvar_0'].dtype == np.float64
        np.testing.assert_array_equal(saved['nvar_0'],result.noise_rms[0].variation.data[0])


def test_builtin_schema_validates_hook_containers():
    from pycwb.constants.user_parameters_schema import schema
    config={'analysis':'2G','ifo':['L1','H1'],'refIFO':'L1',
            'conditioning':{'post_whitening':[{'module':'custom.module','options':{'custom':1}}]}}
    jsonschema.validate(config,schema)
    config['conditioning']['misspelled_stage']=[]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config,schema)


def test_detector_union_and_total_exclusion():
    from pycwb.modules.conditioning_plugins.cwb_gating import apply
    from pycwb.modules.conditioning_plugins.api import HookContext, ConditioningResult
    x=np.zeros(64*128);x[30*128:31*128]=2000
    y=np.zeros(64*128);y[31*128:32*128]=2000
    result=apply(HookContext(SimpleNamespace(segEdge=10),('L1','H1'),None),
                 ConditioningResult([TimeSeries(data=x,t0=1000.,dt=1/128.),
                                     TimeSeries(data=y,t0=1000.,dt=1/128.)],[noise(),noise()]))
    assert result.excluded_intervals==[(1028.,1034.)]
    assert subtract_intervals([(1029,1033)],result.excluded_intervals,1010,1054)==[]


def test_gate_keep_windows_use_circular_livetime():
    from pycwb.types.job import WaveSegment
    # Existing circular-mask semantics must also apply to gates, not just CAT2.
    segment=WaveSegment(index=1,ifos=['L1','H1'],analyze_start=1000,analyze_end=1010,sample_rate=128,seg_edge=0)
    segment._lag_shifts_cache=np.array([[0.,0.],[0.,2.]])
    keep=subtract_intervals(None,[(1004.,1006.)],1000.,1010.)
    assert segment.circular_livetime(0,keep)==8.
    assert segment.circular_livetime(1,keep)==6.


def test_gates_do_not_reapply_cat2_job_duration_cut():
    from pycwb.types.job import WaveSegment
    from pycwb.workflow.subflow.job_segment_veto import _lag_livetime
    segment=WaveSegment(1,['L1','H1'],1000.,1100.,128.,0.)
    context=SimpleNamespace(sub_job_seg=segment,veto_windows=[(1000.,1002.)],
                            selection_exclusions_applied=True,pre_selection_veto_windows=None)
    assert _lag_livetime(context,0)==2.
    assert _lag_livetime(context,0,before_selection=True)==100.

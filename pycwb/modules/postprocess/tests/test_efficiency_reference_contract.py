"""Small exact-count oracles for the public matched-simulation efficiency path."""
import numpy as np
import pandas as pd
import pytest
import xgboost
from pycwb.modules.postprocess import efficiency_metrics as metrics, evaluate

@pytest.fixture
def run_efficiency(tmp_path, monkeypatch):
    # Scores are explicit fixture inputs: exercise selection/counting, not fitting.
    monkeypatch.setattr(xgboost.XGBClassifier, 'load_model', lambda *a: None)
    monkeypatch.setattr(evaluate, '_score_catalog_dataframe', lambda df, *a: df.assign(xgb_prob=df.fixture_score))
    def run(scores, background, cut, duplicate=False):
        n=len(scores)
        df=pd.DataFrame(dict(id=list(range(n))+[None],sim_sim_idx=list(range(n+1)),
            sim_name=['SGE1304Q9']*(n+1),sim_hrss=[1e-22]*(n+1),fixture_score=scores+[0.]))
        if duplicate: df=pd.concat([df,df.iloc[[0]]],ignore_index=True)
        df.to_parquet(tmp_path/'matched.parquet')
        pd.DataFrame({'xgb_prob':background}).to_parquet(tmp_path/'background.parquet')
        return metrics._compute_efficiency_by_waveform_matched(str(tmp_path),'matched.parquet',
            'background.parquet',100.,'unused','bhf',2,None,str(cut),cut,None)
    return run

@pytest.mark.parametrize('background,cut', [([.9,.8,.7],40.),([.9,.8,.8,.7],50.),
    ([.9,.8,.7],20.),([.9,.8,.7],200.),([.9,.8,.7],50.)])
def test_exact_empirical_tail_including_ties_and_gaps(run_efficiency,background,cut):
    scores=[.99,.9,.85,.8,.75,.7,.2]
    expected=sum(sum(b>=s for b in background)/100. <= 1/cut for s in scores)
    result=run_efficiency(scores,background,cut)['efficiency_by_waveform'][0]
    assert result['n_detected']==expected
    assert result['n_total']==8  # includes the missed injection
    assert result['eff_detected']==expected/8

def test_duplicate_recoveries_require_explicit_unique_selection(run_efficiency):
    with pytest.raises(ValueError,match='unique'):
        run_efficiency([.95],[.9],100.,duplicate=True)

@pytest.mark.parametrize('name,expected',[('SGE1304Q9',(9,1304)),('GA0.001',(0,0)),
    ('WNB150_100_0.1',(0,0)),('SG_Q9_235Hz',(9,235))])
def test_actual_campaign_waveform_names(name,expected):
    assert metrics._parse_waveform_q_frequency(name)==expected

@pytest.mark.parametrize('eff',[[],[dict(hrss=1.,efficiency=0.),dict(hrss=2.,efficiency=.2)],
    [dict(hrss=1.,efficiency=.8),dict(hrss=2.,efficiency=1.)]])
def test_unmeasured_crossing_is_not_reported_as_hrss50(eff):
    assert metrics._interpolate_hrss50(eff) is None


def test_cwb_rho1_unique_selection_preserves_trials_and_misses(tmp_path):
    from pycwb.modules.catalog.matching import match_simulations_parquet
    # Same GPS in two trials and a missed third trial: never cross-match trials.
    triggers=pd.DataFrame(dict(id=['a','b','c'],job_id=[1]*3,trial_idx=[0,0,1],
        gps_time=[100.]*3,rho=[20.,10.,8.],rho_alt=[5.,9.,7.]))
    sims=pd.DataFrame(dict(sim_idx=[0,1,2],job_id=[1]*3,trial_idx=[0,1,2],
        gps_time=[100.]*3,real_start=[99.]*3,real_end=[101.]*3))
    triggers.to_parquet(tmp_path/'triggers.parquet');sims.to_parquet(tmp_path/'sims.parquet')
    out=match_simulations_parquet(str(tmp_path/'triggers.parquet'),str(tmp_path/'sims.parquet'),
        how='right',ranking_par='rho_alt').to_pandas().set_index('sim_sim_idx')
    assert out.loc[0,'id']=='b'  # cWB pp_irho=1, not the largest rho[0]
    assert out.loc[1,'id']=='c'
    assert pd.isna(out.loc[2,'id'])
    assert len(out)==3

@pytest.mark.parametrize('scores',[[float('nan'),float('inf'),-.1],[.9,.8,.7]])
def test_invalid_scores_never_detected_and_threshold_matches_tail(scores):
    got,threshold=metrics._empirical_probability_detection(scores,[.9,.8,.8],100.,50.)
    expected=[np.isfinite(s) and sum(b>=s for b in [.9,.8,.8])/100<=1/50 for s in scores]
    np.testing.assert_array_equal(got,expected)
    np.testing.assert_array_equal(got,np.isfinite(scores)&(np.array(scores)>=threshold))

@pytest.mark.parametrize('values,status', [([0.,0.,.2],'above_sampled_range'),
    ([.8,1.,1.],'below_sampled_range')])
def test_unbracketed_sigmoid_is_a_bound_without_calling_fitter(values,status):
    def forbidden(*a):
        pytest.fail('An unbracketed curve must not be reported as a measured hrss50')
    result=metrics._fit_efficiency_curve('SGE1304Q9',[
        dict(hrss=h,efficiency=e) for h,e in zip([1e-22,2e-22,3e-22],values)],
        forbidden,forbidden,forbidden)
    assert result['status']==status
    assert result['hrss50'] is None


def test_known_log_amplitude_crossing():
    assert metrics._interpolate_hrss50([dict(hrss=1e-22,efficiency=.25),
        dict(hrss=4e-22,efficiency=.75)])==pytest.approx(2e-22,rel=1e-14,abs=0)


def test_unique_selection_precedes_reference_time_cut(tmp_path):
    from pycwb.modules.catalog.matching import match_simulations_parquet
    triggers=pd.DataFrame(dict(id=['loud_late','quiet_near'],job_id=[1,1],trial_idx=[0,0],
        gps_time=[200.2,200.01],rho=[30.,25.],rho_alt=[20.,15.]))
    sims=pd.DataFrame(dict(sim_idx=[0],job_id=[1],trial_idx=[0],gps_time=[200.],
        real_start=[199.],real_end=[201.]))
    triggers.to_parquet(tmp_path/'t.parquet');sims.to_parquet(tmp_path/'s.parquet')
    out=match_simulations_parquet(str(tmp_path/'t.parquet'),str(tmp_path/'s.parquet'),
        how='right',ranking_par='rho_alt').to_pandas()
    assert out.id.tolist()==['loud_late']
    # Retain the injection in the denominator, but mark this unique winner missed.
    recovered=out.id.notna() & ((out.gps_time-out.sim_gps_time).abs()<=.1)
    assert len(out)==1 and recovered.sum()==0


def test_public_entrypoints_keep_missed_injections_by_default(tmp_path,monkeypatch):
    from pycwb.modules.postprocess import plot_efficiency as public
    monkeypatch.setattr(xgboost.XGBClassifier,'load_model',lambda *a:None)
    monkeypatch.setattr(evaluate,'_score_catalog_dataframe',lambda df,*a:df.assign(xgb_prob=df.fixture_score))
    matched=pd.DataFrame(dict(id=['a',None,'b',None,'c','d'],sim_sim_idx=range(6),
        sim_name=['SGE1304Q9']*6,sim_hrss=[1e-22,1e-22,2e-22,2e-22,4e-22,4e-22],
        fixture_score=[.1,0,.95,0,.99,.98],xgb_prob=[.1,1.,.95,1.,.99,.98]))
    matched.to_parquet(tmp_path/'matched.parquet')
    pd.DataFrame({'xgb_prob':[.9,.8,.7]}).to_parquet(tmp_path/'bkg.parquet')
    args=dict(work_dir=str(tmp_path),sim_catalog='not_read.parquet',matched_file='matched.parquet',
        bkg_catalog='bkg.parquet',livetime=100.,model_file='unused',ifar='50')
    by_wave=public.compute_efficiency_by_waveform(**args)['efficiency_by_waveform'][0]
    assert (by_wave['n_total'],by_wave['n_detected'])==(6,3)
    curve=public.compute_efficiency_vs_hrss_by_waveform(**args)['curves'][0]['data']
    assert [q['n_total'] for q in curve]==[2,2,2]
    assert [q['efficiency'] for q in curve]==[0,.5,1.]
    # Even a missed row carrying a spurious high score must not become recovered.
    result=public.compute_hrss50(str(tmp_path),'not_read.parquet','bkg.parquet',100.,'50',
        matched_right_file='matched.parquet')
    assert result['hrss50']==pytest.approx(2e-22,rel=1e-14,abs=0)
    assert [q['efficiency'] for q in result['efficiency_curve']]==[0,.5,1.]
    csv_args={k:v for k,v in args.items() if k!='ifar'}
    table=public.compute_hrss50_by_waveform_csv(**csv_args,ifars='50')['hrss50_csv']
    assert len(table)==1 and table[0]['waveform']=='SGE1304Q9'
    with pytest.raises(ValueError,match='denominator'):
        public.compute_efficiency_by_waveform(**args,use_unique_sim=False)
    with pytest.raises(ValueError,match='matched_right_file'):
        public.compute_hrss50(str(tmp_path),'not_read.parquet','bkg.parquet',100.,'50')

@pytest.mark.parametrize('invalid',['typo','-1','0','nan','inf'])
def test_invalid_ifar_does_not_silently_become_one_year(invalid):
    with pytest.raises(ValueError):metrics._parse_ifar_seconds(invalid)

def test_scientific_notation_ifar_seconds():
    assert metrics._parse_ifar_seconds('1e2')==100.

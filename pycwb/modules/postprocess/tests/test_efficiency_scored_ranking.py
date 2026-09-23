"""Regression coverage for efficiency scored ranking."""
from pathlib import Path
import pandas as pd
import pytest
import yaml
from pycwb.modules.postprocess import efficiency_metrics as metrics, plot_efficiency


@pytest.fixture
def ranked_catalogs(tmp_path):
    # Probability orders a ahead of b; rhor orders b ahead of a. c failed the
    # scoring prediction cuts, and None is a missed injection. Neither can pass.
    pd.DataFrame(dict(id=['a', 'b', 'c', None], sim_sim_idx=range(4),
        sim_name=['SG235Q9']*4, sim_hrss=[1e-22]*4,
        xgb_prob=[.99, .2, 1., 1.], rhor=[1., 30., 100., 100.])).to_parquet(tmp_path/'matched.parquet')
    pd.DataFrame(dict(id=['b', 'a', 'unused'], xgb_prob=[.2, .99, 1.],
        rhor=[30., 1., 100.])).to_parquet(tmp_path/'scores.parquet')
    pd.DataFrame(dict(xgb_prob=[.9, .8], rhor=[20., 10.])).to_parquet(tmp_path/'background.parquet')
    return dict(work_dir=str(tmp_path), sim_catalog='unused', matched_file='matched.parquet',
        bkg_catalog='background.parquet', livetime=100., model_file='must_not_load',
        scored_file='scores.parquet', ranking_par='rhor', ifar='100yr')



def test_all_efficiency_entrypoints_share_scored_ranking_and_cuts(ranked_catalogs):
    args=ranked_catalogs
    result=plot_efficiency.compute_efficiency_vs_hrss_by_waveform(**args)
    row=result['curves'][0]['data'][0]
    assert (row['n_total'], row['n_detected']) == (4, 1)
    assert result['ranking_par'] == 'rhor'
    assert result['prob_threshold'] is None
    assert result['ranking_threshold'] > 20.
    row=plot_efficiency.compute_efficiency_by_waveform(**args)['efficiency_by_waveform'][0]
    assert (row['n_total'], row['n_detected']) == (4, 1)
    scalar={k:v for k,v in args.items() if k!='matched_file'}
    result=plot_efficiency.compute_hrss50(**scalar, matched_right_file=args['matched_file'])
    assert result['efficiency_curve'][0]['efficiency'] == .25
    csv_args={k:v for k,v in args.items() if k!='ifar'}
    row=plot_efficiency.compute_hrss50_by_waveform_csv(**csv_args, ifars='100yr')['hrss50_csv'][0]
    assert row['ranking_par']=='rhor' and row['ranking_threshold'] > 20.



def test_duplicate_scores_rejected(ranked_catalogs):
    path=Path(ranked_catalogs['work_dir'])/'scores.parquet'
    scores=pd.read_parquet(path)
    pd.concat([scores, scores.iloc[[0]]]).to_parquet(path)
    with pytest.raises(ValueError, match='unique'):
        plot_efficiency.compute_efficiency_by_waveform(**ranked_catalogs)



def test_model_scoring_preserves_prediction_cut_rejections(tmp_path, monkeypatch):
    import xgboost
    from pycwb.modules.postprocess import evaluate
    monkeypatch.setattr(xgboost.XGBClassifier, 'load_model', lambda *a: None)
    monkeypatch.setattr(evaluate, '_score_catalog_dataframe',
        lambda df, *args: df[df.id=='b'].assign(rhor=30.))
    matched=pd.DataFrame({'id':['a','b',None]})
    scores=metrics._matched_ranking_scores(matched, str(tmp_path), 'rhor', model_file='unused')
    assert scores.isna().tolist()==[True,False,True]
    assert scores.iloc[1]==30.



def test_yaml_ranking_workflow_writes_report(ranked_catalogs):
    from pycwb.post_production.workflow import run_workflow
    p=Path(ranked_catalogs['work_dir'])
    args={k:v for k,v in ranked_catalogs.items() if k!='work_dir'}
    args['output_file']='efficiency.png'
    args['fit_parameters_file']='efficiency.csv'
    workflow={'vars':{'work_dir':str(p)}, 'steps':[
        {'id':'efficiency','action':'postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform','args':args},
        {'id':'report','action':'postprocess.report_builder.postproduction_report','args':{
            'workflow_file':'workflow.yaml','production_catalog_file':'matched.parquet',
            'output_file':'report/index.html','title':'Manual ranking regression',
            'simulation_runs':[{'label':'Native rhor','matched_file':'matched.parquet',
                'scored_catalog':'scores.parquet','plots':[{'path':'efficiency.png','label':'rhor sensitivity'}]}]}}
    ]}
    (p/'workflow.yaml').write_text(yaml.safe_dump(workflow))
    run_workflow(str(p/'workflow.yaml'), generate_diagram=False)
    assert (p/'report/index.html').exists()
    assert (p/'efficiency.png').stat().st_size>0
    assert pd.read_csv(p/'efficiency.csv').ranking_par.tolist()==['rhor']

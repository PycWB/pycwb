"""Regression coverage for far ties."""
import pandas as pd


def test_far_table_uses_inclusive_ties(tmp_path, monkeypatch):
    import xgboost
    from pycwb.modules.postprocess import evaluate
    monkeypatch.setattr(xgboost.XGBClassifier,'load_model',lambda *a:None)
    monkeypatch.setattr(evaluate,'_scoring_read_columns',lambda *a,**k:None)
    monkeypatch.setattr(evaluate,'_score_catalog_dataframe',lambda df,*a:df)
    pd.DataFrame(dict(id=['a','b','c'],rhor=[20.,20.,10.],xgb_prob=[.9,.8,.7])).to_parquet(tmp_path/'bkg.parquet')
    result=evaluate.evaluate_far_rho(str(tmp_path),'bkg.parquet','unused',livetime=100.,
                                   ranking_par='rhor',exclude_zero_lag=False)
    assert [row['n_events'] for row in result['far_rho']]==[2,2,3]
    assert [row['far'] for row in result['far_rho']]==[.02,.02,.03]

"""cWB field semantics must survive the Trigger catalog representation."""
import numpy as np
import pandas as pd
import pytest

from pycwb.modules.cwb_xgboost.config import xgb_config
from pycwb.modules.cwb_xgboost.read_data import preprocess_events
from pycwb.modules.postprocess.train_xgboost import (
    _map_catalog_columns, _xgb_required_input_columns, _read_and_concat,
)


@pytest.mark.parametrize('order', [('L1', 'H1'), ('H1', 'L1')])
def test_catalog_features_match_cwb_fields_after_projection(tmp_path, order):
    # Values distinguish norm from ECOR and signal energy from signal/data.
    catalog = dict(ifo_list=list(order), coherent_energy=100.,
                   coherent_energy_norm=999., packet_norm=9., penalty=1.,
                   likelihood=100., net_cc=.9, q_veto=4., q_factor=6.,
                   rho=8., Lveto2=0.)
    flat = dict(ecor=100., norm=9., penalty=1., likelihood=100.,
                netcc0=.9, Qveto0=4., Qveto1=6., rho0=8., Lveto2=0.)
    for i, ifo in enumerate(order):
        for source, root, value in [('signal_energy','sSNR',40.+20*i),
                                    ('data_energy','snr',80.+40*i),
                                    ('noise_rms','noise',np.float32((2.+i)*1e-24)),
                                    ('central_freq','frequency',100.+50*i),
                                    ('duration','duration',.1+i),
                                    ('bandwidth','bandwidth',20.+i)]:
            catalog[f'{source}_{ifo}'] = value
            flat[f'{root}{i}'] = value
    path=tmp_path/'catalog.parquet'
    pd.DataFrame([catalog]).to_parquet(path,index=False)
    _, features, caps, balance, options=xgb_config('blf',2)
    columns=_xgb_required_input_columns([str(path)],2,options,caps,balance,features)
    mapped=_map_catalog_columns(_read_and_concat([str(path)],'test',columns),2)
    actual=preprocess_events(mapped,2,options,caps)
    expected=preprocess_events(pd.DataFrame([flat]),2,options,caps)
    features=list(dict.fromkeys(features+['mSNR/likelihood','ecor/likelihood','noise']))
    np.testing.assert_allclose(actual[features],expected[features],rtol=0,atol=0)
    assert actual.loc[0,'mSNR/likelihood']==.4
    assert actual.loc[0,'norm']==9.
    assert 0 < actual.loc[0,'noise'] < 2e-24
    assert _map_catalog_columns(pd.DataFrame([catalog]),2).loc[0,'frequency0']==100.


def test_mixed_detector_order_fails_explicitly():
    with pytest.raises(ValueError,match='consistent detector order'):
        _map_catalog_columns(pd.DataFrame({'ifo_list':[['L1','H1'],['H1','L1']]}),2)


def test_ecor_is_not_a_fallback_for_missing_packet_norm():
    mapped=_map_catalog_columns(pd.DataFrame({'coherent_energy_norm':[999.]}),2)
    assert 'norm' not in mapped

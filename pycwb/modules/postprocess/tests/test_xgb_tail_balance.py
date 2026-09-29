import pandas as pd
import pytest

from pycwb.modules.cwb_xgboost.utils_extended import get_balanced_tail


def test_signal_tail_upsampling_preserves_background_and_row_order():
    events=pd.DataFrame({'event':[0,1,2,3,4,5],
                         'rho0_40d0':[8.,40.,40.,9.,40.,40.],
                         'classifier':[1,0,1,0,0,0]})
    result=get_balanced_tail(events,{'rho0':40},150914)
    assert result.event.tolist()==[0,1,3,4,5,2,2,2]
    assert result.loc[result.classifier==0,'event'].tolist()==[1,3,4,5]


def test_only_signal_tail_is_downsampled():
    events=pd.DataFrame({'event':[0,1,2,3,4],
                         'rho0_40d0':[40.,40.,8.,40.,40.],
                         'classifier':[1,0,0,1,1]})
    result=get_balanced_tail(events,{'rho0':40},150914)
    assert result.event.iloc[:2].tolist()==[1,2]
    assert len(result)==3 and result.classifier.sum()==1
    assert result.event.iloc[-1] in [0,3,4]


def test_missing_signal_tail_does_not_silently_drop_loud_background():
    events=pd.DataFrame({'rho0_40d0':[8.,40.],'classifier':[1,0]})
    with pytest.raises(ValueError,match='without signal tail'):
        get_balanced_tail(events,{'rho0':40},150914)

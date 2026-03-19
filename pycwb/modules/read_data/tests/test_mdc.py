import pytest
import numpy as np
from unittest.mock import MagicMock
from pycwb.types.time_series import TimeSeries
from pycwb.modules.read_data.mdc import generate_strain_from_injection


def _make_ts():
    """Create a minimal TimeSeries for testing."""
    return TimeSeries(data=np.zeros(100), t0=0.0, dt=1.0 / 4096)


def test_generator_in_injection(mocker):
    # Setup injection with specific generator
    injection = {
        'generator': 'custom_module.waveform_generator',
        'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234,
    }
    config = MagicMock(injection={})
    sample_rate = 4096
    ifos = ['H1', 'L1']

    h1_ts, l1_ts = _make_ts(), _make_ts()
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'strain', 'L1': l1_ts, 'H1': h1_ts})
    strains = generate_strain_from_injection(injection, config, sample_rate, ifos)

    assert strains[0] is not None and strains[1] is not None

    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'strain', 'L1': l1_ts, 'H1': h1_ts, 'V1': _make_ts()})
    with pytest.raises(ValueError, match="ifos"):
        generate_strain_from_injection(injection, config, sample_rate, ifos)
    


def test_generator_in_config(mocker):
    injection = {'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234}
    config = MagicMock(injection={'generator': 'config_module.config_func'})
    sample_rate = 4096
    ifos = ['H1', 'L1']
    
    h1_ts, l1_ts = _make_ts(), _make_ts()
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'strain', 'H1': h1_ts, 'L1': l1_ts})
    
    strains = generate_strain_from_injection(injection, config, sample_rate, ifos)
    
    assert len(strains) == 2


def test_default_generator_warning(mocker):
    injection = {
        'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234, 
    }

    config = MagicMock(injection={})
    ifos = ['H1', 'L1']
    
    hp_ts, hc_ts = _make_ts(), _make_ts()
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'polarizations', 'hp': hp_ts, 'hc': hc_ts})
    mocker.patch('pycwb.modules.read_data.mdc.project_to_detector', return_value=[_make_ts()])

    with pytest.warns(DeprecationWarning, match="waveform generator"):
        generate_strain_from_injection(injection, config, 4096, ifos)


def test_tuple_return_backward_compat(mocker):
    injection = {
        'ra': 0.1, 'dec': 0.2, 'pol': 0.3, 'gps_time': 1234,
        'generator': 'custom_module.waveform_generator',
    }
    config = MagicMock(injection={})
    
    hp_ts, hc_ts = _make_ts(), _make_ts()
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: (hp_ts, hc_ts))
    mocker.patch('pycwb.modules.read_data.mdc.project_to_detector', return_value=[_make_ts(), _make_ts()])

    with pytest.warns(DeprecationWarning, match="tuple"):
        strains = generate_strain_from_injection(injection, config, 4096, ['H1', 'L1'])
    
    assert len(strains) == 2


def test_strain_type_return(mocker):
    injection = {        
        'generator': 'custom_module.waveform_generator', 
        'delta_t': 1/4096
    }
    config = MagicMock(injection={})
    ifos = ['H1', 'V1']
    h1_ts, v1_ts = _make_ts(), _make_ts()
    mock_strain = {'type': 'strain', 'H1': h1_ts, 'V1': v1_ts}
    
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: mock_strain)
    
    strains = generate_strain_from_injection(injection, config, 4096, ifos)
    assert len(strains) == 2


def test_polarizations_projection(mocker):
    injection = {
        'generator': 'mymod.f',
        'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234
    }
    config = MagicMock(injection={})
    ifos = ['H1']
    
    mock_hp, mock_hc = MagicMock(), MagicMock()
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'polarizations', 'hp': mock_hp, 'hc': mock_hc})
    project_to_detector = mocker.patch('pycwb.modules.read_data.mdc.project_to_detector', return_value=['strain'])
    
    strains = generate_strain_from_injection(injection, config, 4096, ifos)
    assert len(strains) == 1
    # Verify project_to_detector called with correct params
    args, _ = project_to_detector.call_args
    assert args[2] == injection['ra']
    assert args[3] == injection['dec']


def test_missing_parameters_error(mocker):
    injection = {'generator': 'mymod.f', 'ra': None, 'dec': 0.2}
    config = MagicMock(injection={})
    
    hp_ts, hc_ts = _make_ts(), _make_ts()
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'polarizations', 'hp': hp_ts, 'hc': hc_ts})

    with pytest.raises(ValueError, match="ra"):
        generate_strain_from_injection(injection, config, 4096, ['H1'])


def test_mismatched_ifos_error(mocker):
    injection = {'generator': 'mymod.f'}
    config = MagicMock(injection={})

    # for missing ifos
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'strain', 'H1': _make_ts()})
    with pytest.raises(ValueError, match="ifos"):
        generate_strain_from_injection(injection, config, 4096, ['H1', 'L1'])

    # for extra ifos
    mocker.patch('pycwb.modules.read_data.mdc.import_function', return_value=lambda **kwargs: {'type': 'strain', 'H1': _make_ts(), 'L1': _make_ts(), 'V1': _make_ts()})
    with pytest.raises(ValueError, match="ifos"):
        generate_strain_from_injection(injection, config, 4096, ['H1', 'L1'])
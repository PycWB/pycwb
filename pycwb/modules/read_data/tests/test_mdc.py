import pytest
from unittest.mock import MagicMock, patch
from pycwb.modules.read_data.mdc import generate_strain_from_injection

def test_generator_in_injection(mocker):
    # Setup injection with specific generator
    injection = {
        'generator': {'module': 'custom_module', 'function': 'waveform_generator'},
        'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234,
    }
    config = MagicMock(injection={})
    sample_rate = 4096
    ifos = ['H1', 'L1']

    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', lambda x: x)

    with patch('pycwb.modules.read_data.mdc.import_helper') as mock_import:
        mock_module = MagicMock()
        mock_module.waveform_generator.return_value = {'type': 'strain', 'L1': 'l1_strain', 'H1': 'h1_strain'}
        mock_import.return_value = mock_module
    
        # Execute
        strains = generate_strain_from_injection(injection, config, sample_rate, ifos)
    
        # Assert
        assert strains == ['h1_strain', 'l1_strain']
        mock_module.waveform_generator.assert_called_once_with(**injection)

    with patch('pycwb.modules.read_data.mdc.import_helper') as mock_import:
        mock_module = MagicMock()
        mock_module.waveform_generator.return_value = {'type': 'strain', 'L1': 'l1_strain', 'H1': 'h1_strain', 'V1': 'v1_strain'}
        mock_import.return_value = mock_module
    
        # Execute
        with pytest.raises(ValueError, match="ifos"):
            generate_strain_from_injection(injection, config, sample_rate, ifos)
    


def test_generator_in_config(mocker):
    injection = {'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234}
    config = MagicMock(injection={'generator': {'module': 'config_module', 'function': 'config_func'}})
    sample_rate = 4096
    ifos = ['H1', 'L1']
    
    # Mock generator
    mock_module = MagicMock()
    mock_module.config_func.return_value = {'type': 'strain', 'H1': 'H1-strain', 'L1': 'L1-strain'}
    mocker.patch('pycwb.modules.read_data.mdc.import_helper', return_value=mock_module)
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', lambda x: x)
    
    strains = generate_strain_from_injection(injection, config, sample_rate, ifos)
    
    assert strains == ['H1-strain', 'L1-strain']
    mock_module.config_func.assert_called_once()


def test_default_generator_warning(mocker):
    injection = {
        'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234, 
    }

    config = MagicMock(injection={})
    ifos = ['H1', 'L1']
    
    # Mock pycbc waveform
    mocker.patch('pycbc.waveform.get_td_waveform', return_value=([], []))
    mocker.patch('pycwb.modules.read_data.mdc.project_to_detector', return_value=['strain'])
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', lambda x: x)

    with pytest.warns(DeprecationWarning, match="waveform generator"):
        generate_strain_from_injection(injection, config, 4096, ifos)


def test_tuple_return_backward_compat(mocker):
    injection = {
        'ra': 0.1, 'dec': 0.2, 'pol': 0.3, 'gps_time': 1234,
        'generator': {'module': 'custom_module', 'function': 'waveform_generator'},
    }
    config = MagicMock(injection={})
    
    # Mock tuple return
    mock_module = MagicMock()
    mock_module.waveform_generator.return_value = ('hp', 'hc')
    mocker.patch('pycwb.modules.read_data.mdc.import_helper', return_value=mock_module)
    mocker.patch('pycwb.modules.read_data.mdc.project_to_detector', return_value=['h1', 'l1'])
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', lambda x: x)

    with pytest.warns(DeprecationWarning, match="tuple"):
        strains = generate_strain_from_injection(injection, config, 4096, ['H1', 'L1'])
    
    assert strains == ['h1', 'l1']


def test_strain_type_return(mocker):
    injection = {        
        'generator': {'module': 'custom_module', 'function': 'waveform_generator'}, 
        'delta_t': 1/4096
    }
    config = MagicMock(injection={})
    ifos = ['H1', 'V1']
    mock_strain = {'type': 'strain', 'H1': 'h1', 'V1': 'v1'}
    
    mock_module = MagicMock()
    mock_module.waveform_generator.return_value = mock_strain
    mocker.patch('pycwb.modules.read_data.mdc.import_helper', return_value=mock_module)
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', lambda x: x)
    
    strains = generate_strain_from_injection(injection, config, 4096, ifos)
    assert strains == ['h1', 'v1']


def test_polarizations_projection(mocker):
    injection = {
        'generator': {'module': 'm', 'function': 'f'},
        'ra': 0.5, 'dec': 0.2, 'pol': 1.0, 'gps_time': 1234
    }
    config = MagicMock(injection={})
    ifos = ['H1']
    
    mock_hp, mock_hc = MagicMock(), MagicMock()
    mock_module = MagicMock()
    mock_module.f.return_value = {'type': 'polarizations', 'hp': mock_hp, 'hc': mock_hc}
    mocker.patch('pycwb.modules.read_data.mdc.import_helper', return_value=mock_module)
    project_to_detector = mocker.patch('pycwb.modules.read_data.mdc.project_to_detector', return_value=['strain'])
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', side_effect=lambda x: x)
    
    strains = generate_strain_from_injection(injection, config, 4096, ifos)
    assert len(strains) == 1
    # Verify project_to_detector called with correct params
    args, _ = project_to_detector.call_args
    assert args[2] == injection['ra']
    assert args[3] == injection['dec']


def test_missing_parameters_error(mocker):
    injection = {'generator': {'module': 'm', 'function': 'f'}, 'ra': None, 'dec': 0.2}
    config = MagicMock(injection={})
    
    mock_module = MagicMock()
    mock_module.f.return_value = {'type': 'polarizations', 'hp': 'hp', 'hc': 'hc'}
    mocker.patch('pycwb.modules.read_data.mdc.import_helper', return_value=mock_module)
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', side_effect=lambda x: x)

    with pytest.raises(ValueError, match="ra"):
        generate_strain_from_injection(injection, config, 4096, ['H1'])


def test_mismatched_ifos_error(mocker):
    injection = {'generator': {'module': 'm', 'function': 'f'}}
    config = MagicMock(injection={})
    
    mock_module = MagicMock()
    mocker.patch('pycwb.modules.read_data.mdc.import_helper', return_value=mock_module)
    mocker.patch('pycwb.modules.read_data.mdc.convert_to_pycbc_timeseries', side_effect=lambda x: x)

    # for missing ifos
    mock_strain = {'type': 'strain', 'H1': 'h1'}
    mock_module.f.return_value = mock_strain
    with pytest.raises(ValueError, match="ifos"):
        generate_strain_from_injection(injection, config, 4096, ['H1', 'L1'])

    # for extra ifos
    mock_strain = {'type': 'strain', 'H1': 'h1', 'L1': 'l1', 'V1': 'v1'}
    mock_module.f.return_value = mock_strain
    with pytest.raises(ValueError, match="ifos"):
        generate_strain_from_injection(injection, config, 4096, ['H1', 'L1'])
"""Regression coverage for ifar duration."""
import pytest
from pycwb.modules.postprocess import efficiency_metrics as metrics


@pytest.mark.parametrize('label,seconds', [('100yr', 3155760000.), ('0.5yr', 15778800.),
    ('2wk', 1209600.), ('1mo', 2592000.), ('6mo', 15778800.), ('1e2', 100.), (100, 100.)])
def test_ifar_duration_units(label, seconds):
    assert metrics._parse_ifar_seconds(label) == seconds



@pytest.mark.parametrize('label', ['100years', '0yr', '-1yr', 'nan', 'inf', '1yrgarbage'])
def test_ifar_duration_rejects_invalid_labels(label):
    with pytest.raises(ValueError):
        metrics._parse_ifar_seconds(label)

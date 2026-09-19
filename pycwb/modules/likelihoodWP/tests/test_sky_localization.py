"""Localization probabilities, deciles and release-oracle contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
from pycwb.modules.likelihoodWP.sky_localization import localize_sky

REFERENCE = Path(__file__).parent / 'reference'
CASES = json.loads((REFERENCE / 'sky_localization_cases.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_installed_release_oracle(case):
    with np.load(REFERENCE / 'sky_localization_release.npz') as data:
        name = case['name']
        result = localize_sky(data[name+'__statistic'], data[name+'__antenna'],
                              case['rms'], use_prior=case['prior'], n_sky=case['n_sky'])
        assert result is not None
        np.testing.assert_allclose(result.probability, data[name+'__probability'],
                                   rtol=2e-14, atol=1e-17)
        np.testing.assert_array_equal(result.indices, data[name+'__indices'])
        np.testing.assert_array_equal(result.error_regions, data[name+'__areas'])


@pytest.mark.parametrize('scale', [0., np.nan, np.inf])
def test_invalid_scale_is_unavailable(scale):
    assert localize_sky(np.ones(12288), np.ones(12288), scale) is None


def test_no_positive_sky_is_unavailable():
    assert localize_sky(np.zeros(12288), np.ones(12288), 1.) is None


def test_invalid_pixel_cannot_become_reconstructed_peak():
    statistic = np.linspace(1., 20., 12288)
    statistic[-1] = np.nan
    result = localize_sky(statistic, np.ones(12288), 1.)
    assert result.indices[0] == 12286
    assert result.probability[-1] == 0.
    assert np.isclose(result.probability.sum(), 1.)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match='aligned vectors'):
        localize_sky(np.ones(12288), np.ones(3), 1.)

"""Release pointer-sort fixtures, including tied values and permuted inputs."""
from pathlib import Path
import json
import numpy as np
import pytest
from pycwb.modules.likelihoodWP.sky_order import wave_sort_indices

CASES = json.loads((Path(__file__).parent / 'reference/sky_sort_oracle.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda r: str(len(r['values'])))
def test_release_sort_permutation(case):
    original = np.asarray(case['initial'])
    result = wave_sort_indices(case['values'], original)
    np.testing.assert_array_equal(result, case['expected'])
    np.testing.assert_array_equal(original, case['initial'])
    values = np.asarray(case['values'])[result]
    assert np.all(values[1:] >= values[:-1])


def test_ld_tied_posterior_mode_matches_release():
    from pycwb.modules.likelihoodWP.sky_localization import localize_sky
    with np.load(Path(__file__).parent / 'reference/ld_sky_tie_oracle.npz') as case:
        result = localize_sky(case['statistic'], case['antenna'], case['scale'], use_prior=True)
        assert result.indices[0] == int(case['selected_index'])
        np.testing.assert_array_equal(result.error_regions, case['areas'])
        np.testing.assert_allclose(result.probability, case['probability'], rtol=1e-13, atol=0.)

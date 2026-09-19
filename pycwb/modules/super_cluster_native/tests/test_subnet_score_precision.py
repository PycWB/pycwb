"""Release-input regression: rounding the ranking score changes a near-tied sky choice."""
from pathlib import Path
import numpy as np
import pytest
from pycwb.modules.super_cluster_native.sub_net_cut import (
    optimze_sky_loc, optimze_sky_loc_from_td, mra_statistics_from_td,
)


@pytest.mark.parametrize('precomputed_energy', [False, True])
def test_reference_subnet_score_rounding_preserves_sky_and_cut(precomputed_energy):
    # Captured cWB 6.4.6.9, Chunk 16a LF job 1, lag 209. The original
    # double-promoted AA selected sky 60 and accepted; cWB selects 37 and rejects.
    with np.load(Path(__file__).parent/'reference/subnet_reference_209.npz') as a:
        n_pix = a['rms'].shape[0]
        scan_function = optimze_sky_loc if precomputed_energy else optimze_sky_loc_from_td
        energy = (a['td00'] ** 2 + a['td90'] ** 2,) if precomputed_energy else ()
        scan = scan_function(
            2, n_pix, len(a['FP']), a['FP'], a['FX'], a['rms'],
            a['td00'], a['td90'], *energy, a['ml'], np.float32(a['network_energy_threshold']),
            float(a['e2or']), float(a['subcut']),
        )
        assert scan[0] == int(a['reference_sky_index']) == 37
        mra = mra_statistics_from_td(
            2, n_pix, a['FP'], a['FX'], a['rms'], a['td00'], a['td90'],
            a['ml'], np.float32(a['network_energy_threshold']), float(a['e2or']),
            float(a['subcut']), a['xtalk'], a['lookup'], scan[0],
        )
        assert mra[1] < float(a['subrho'])
        np.testing.assert_allclose(mra[1], 4.495407036547879, rtol=0, atol=1e-6)

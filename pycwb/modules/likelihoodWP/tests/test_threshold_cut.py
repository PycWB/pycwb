"""Characterization tests for threshold_cut behavior.

Covers 2G and XGB rejection conditions using small SkyStatistics fixtures.
"""

import numpy as np
import pytest


def _make_sky_statistics(**overrides):
    """Build a minimal SkyStatistics dataclass for threshold testing."""
    from pycwb.modules.likelihoodWP.typing import SkyStatistics

    defaults = dict(
        Gn=np.float32(0.1),
        Ec=np.float32(5.0),
        Dc=np.float32(0.5),
        Rc=np.float32(0.8),
        Eh=np.float32(0.2),
        Es=np.float32(0.1),
        Np=np.float32(3.0),
        Em=np.float32(6.0),
        Lm=np.float32(4.0),
        norm=np.float32(1.5),
        cc=np.float32(2.0),
        rho=np.float32(8.0),
        xrho=np.float32(7.5),
        Lo=np.float32(3.0),
        Eo=np.float32(10.0),
        N_pix_effective=np.float32(5.0),
        energy_array_plus=np.zeros(3, dtype=np.float32),
        energy_array_cross=np.zeros(3, dtype=np.float32),
        v00=np.zeros((2, 3), dtype=np.float32),
        v90=np.zeros((2, 3), dtype=np.float32),
        pd=np.zeros((2, 3), dtype=np.float32),
        pD=np.zeros((2, 3), dtype=np.float32),
        ps=np.zeros((2, 3), dtype=np.float32),
        pS=np.zeros((2, 3), dtype=np.float32),
        pixel_mask=np.ones(3, dtype=np.int32),
        gaussian_noise_correction=np.zeros(3, dtype=np.float32),
        noise_amplitude_00=np.zeros((2, 3), dtype=np.float32),
        noise_amplitude_90=np.zeros((2, 3), dtype=np.float32),
        coherent_energy=np.zeros(3, dtype=np.float32),
        p00_POL=np.zeros((2, 3), dtype=np.float32),
        p90_POL=np.zeros((2, 3), dtype=np.float32),
        r00_POL=np.zeros((2, 3), dtype=np.float32),
        r90_POL=np.zeros((2, 3), dtype=np.float32),
        S_snr=np.zeros(2, dtype=np.float32),
        f=np.zeros((3, 2), dtype=np.float32),
        F=np.zeros((3, 2), dtype=np.float32),
    )
    defaults.update(overrides)
    # Default valid 2G cluster
    defaults.setdefault("Ec", np.float32(5.0))
    defaults.setdefault("Rc", np.float32(0.8))
    defaults.setdefault("cc", np.float32(2.0))
    return SkyStatistics(**defaults)


class TestThresholdCut:
    """Test threshold_cut behavior for 2G and XGB modes."""

    # ------------------------------------------------------------------
    # 2G mode
    # ------------------------------------------------------------------
    def test_2g_accepted(self):
        """2G mode: healthy cluster passes."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(
            Lm=np.float32(4.0),
            Eo=np.float32(10.0),
            Eh=np.float32(0.2),
            Ec=np.float32(5.0),
            Rc=np.float32(0.8),
            cc=np.float32(2.0),
            N_pix_effective=np.float32(5.0),
        )
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=0.5, xgb_rho_mode=False)
        assert result is None, f"Expected None but got: {result}"

    def test_2g_reject_lm_negative(self):
        """2G mode: Lm <= 0 triggers rejection."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(Lm=np.float32(0.0))
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=0.5, xgb_rho_mode=False)
        assert result is not None
        assert "Lm" in result

    def test_2g_reject_eo_eh_nonpositive(self):
        """2G mode: (Eo - Eh) <= 0 triggers rejection."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(Eo=np.float32(0.1), Eh=np.float32(0.2))
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=0.5, xgb_rho_mode=False)
        assert result is not None
        assert "(Eo - Eh)" in result

    def test_2g_reject_netec_too_low(self):
        """2G mode: Ec * Rc / cc < netEC_threshold triggers rejection."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(
            Ec=np.float32(0.1), Rc=np.float32(0.8), cc=np.float32(2.0))
        # Ec*Rc/cc = 0.04 < 0.5
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=0.5, xgb_rho_mode=False)
        assert result is not None

    def test_2g_reject_n_lt_1(self):
        """2G mode: N < 1 triggers rejection."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(N_pix_effective=np.float32(0.0))
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=0.01, xgb_rho_mode=False)
        assert result is not None
        assert "N" in result

    # ------------------------------------------------------------------
    # XGB mode
    # ------------------------------------------------------------------
    def test_xgb_accepted(self):
        """XGB mode: rho >= |netRHO| passes."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(rho=np.float32(8.0))
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=2.0,
                               net_rho_threshold=5.0, xgb_rho_mode=True)
        assert result is None

    def test_xgb_reject_rho_too_low(self):
        """XGB mode: rho < net_rho_threshold triggers rejection."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(rho=np.float32(2.0))
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=2.0,
                               net_rho_threshold=5.0, xgb_rho_mode=True)
        assert result is not None
        assert "rho" in result

    def test_xgb_reject_nan_rho(self):
        """XGB mode: NaN rho triggers rejection."""
        from pycwb.modules.likelihoodWP.detection_statistics import threshold_cut

        sky = _make_sky_statistics(rho=np.float32(np.nan))
        result = threshold_cut(sky, network_energy_threshold=1.0,
                               netEC_threshold=2.0,
                               net_rho_threshold=5.0, xgb_rho_mode=True)
        assert result is not None

    # ------------------------------------------------------------------
    # Alias identity
    # ------------------------------------------------------------------
    def test_alias_identity(self):
        """threshold_cut and get_likelihood_rejection_reason are the same."""
        from pycwb.modules.likelihoodWP.detection_statistics import (
            threshold_cut, get_likelihood_rejection_reason,
        )
        assert threshold_cut is get_likelihood_rejection_reason

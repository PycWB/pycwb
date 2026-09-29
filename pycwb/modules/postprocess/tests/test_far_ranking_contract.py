"""A FAR lookup must be calibrated on the statistic being thresholded."""

import numpy as np
import pandas as pd
import pytest

from pycwb.modules.postprocess import far


@pytest.mark.parametrize("empty", [False, True])
def test_mismatched_far_statistic_rejected_before_attachment_or_csv(tmp_path, empty):
    frame = pd.DataFrame({"rho": [] if empty else [8.], "xgb_prob": [] if empty else [.9]})
    table = {"ranking_par": "rho", "bins": [5., 10.], "far": [2., 1.]}
    with pytest.raises(ValueError, match="'rho'.*'xgb_prob'"):
        far.attach_far_and_significance(frame, table, "xgb_prob", 100.)
    with pytest.raises(ValueError, match="'rho'.*'xgb_prob'"):
        far.write_loudest_background_triggers(frame, str(tmp_path), "xgb_prob", table)
    assert not (tmp_path / "loudest_background_triggers.csv").exists()


@pytest.mark.parametrize("statistic", ["rho", "xgb_prob", "custom_rank"])
def test_matching_far_statistic_keeps_calibration(statistic, tmp_path):
    frame = pd.DataFrame({statistic: [0.25, .75]})
    table = {"ranking_par": statistic, "bins": [0., .5], "far": [2., 1.]}
    result = far.attach_far_and_significance(frame, table, statistic, 31557600.)
    np.testing.assert_array_equal(result.far_attached, [2., 1.])
    np.testing.assert_array_equal(result.ifar_years, [.5, 1.])
    path, count = far.write_loudest_background_triggers(frame, str(tmp_path), statistic, table)
    assert count == 2
    np.testing.assert_array_equal(pd.read_csv(path).far_attached, [1., 2.])


def test_unlabeled_legacy_table_is_only_valid_for_rho():
    frame = pd.DataFrame({"rho": [8.], "xgb_prob": [.9]})
    table = {"bins": [5., 10.], "far": [2., 1.]}
    assert far.attach_far_and_significance(frame, table, "rho", 100.).far_attached.tolist() == [2.]
    with pytest.raises(ValueError, match="'rho'.*'xgb_prob'"):
        far.attach_far_and_significance(frame, table, "xgb_prob", 100.)

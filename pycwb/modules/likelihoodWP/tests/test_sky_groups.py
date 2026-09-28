import numpy as np
from pycwb.modules.likelihoodWP.sky_delay_groups import make_delay_groups, delay_groups_for_grid


def test_groups_contain_equal_delay_tuples():
    ml = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]])
    order, offsets = make_delay_groups(ml)
    np.testing.assert_array_equal(np.sort(order), np.arange(6))
    for a, b in zip(offsets[:-1], offsets[1:]):
        for i in order[a:b]:
            np.testing.assert_array_equal(ml[:, i], ml[:, order[a]])


def test_cache_grid_identity_and_separate_coarse_geometry():
    setup = {}
    main = np.array([[0, 1, 0], [1, 0, 1]])
    coarse = np.array([[0, 1], [1, 0]])
    order, offsets = delay_groups_for_grid(setup, main)
    assert delay_groups_for_grid(setup, main)[0] is order
    other = delay_groups_for_grid(setup, coarse, True)
    assert len(other[0]) == 2
    assert delay_groups_for_grid(setup, main)[1] is offsets
    replacement = main.copy()
    replacement[:, 0] = 2
    updated = delay_groups_for_grid(setup, replacement)
    assert updated[0] is not order
    for actual, expected in zip(updated, make_delay_groups(replacement)):
        np.testing.assert_array_equal(actual, expected)

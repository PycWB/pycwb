import numpy as np
import pytest
from pycwb.modules.super_cluster_native.utils import get_defragment_link


@pytest.mark.parametrize("gap,expected", [(15.999, 0), (16.0, 1), (20.0, 1)])
def test_frequency_gap_is_in_hz_and_includes_equality(gap, expected):
    # Equal-rate pixels centered at 20 and 40 Hz each have 4 Hz bandwidth.
    # Their edge-to-edge separation is 16 Hz. Feature column 1 stores 2*f.
    pixels = np.array([[0, 40, 0.125, 4, 0, 0, 0], [0, 80, 0.125, 4, 1, 0, 0]], dtype=float)
    links = get_defragment_link(pixels, 0.2, gap, 2)
    assert len(links) == expected
    if expected:
        np.testing.assert_array_equal(links, [[0, 1]])


def test_defragmentation_can_be_disabled_for_overlapping_pixels():
    pixels = np.array([[0, 40, 0.125, 4, 0, 0, 0], [0, 40, 0.125, 4, 1, 0, 0]], dtype=float)
    assert get_defragment_link(pixels, 0.0, 0.0, 2).shape == (0, 2)


def test_transitive_frequency_neighbors_and_empty_input():
    pixels = np.array(
        [[0, 40, 0.125, 4, 0, 0, 0], [0, 80, 0.125, 4, 1, 0, 0], [0, 120, 0.125, 4, 2, 0, 0]], dtype=float
    )
    links = get_defragment_link(pixels, 0.2, 16.0, 2)
    assert set(map(tuple, links)) == {(0, 1), (1, 2)}
    assert get_defragment_link(pixels[:0], 0.2, 16.0, 2).shape == (0, 2)

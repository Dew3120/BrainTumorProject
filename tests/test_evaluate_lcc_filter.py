import evaluate_lcc_filter as lcc
import numpy as np


def test_largest_component_uses_eight_connectivity_and_top_left_tie_break():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[3, 3] = True
    mask[5:7, 5:7] = True
    mask[4, 6] = True
    filtered, removed_components, removed_pixels = lcc.keep_largest_component(mask)
    assert removed_components == 1
    assert removed_pixels == 5
    assert filtered[1, 1]
    assert not filtered[6, 6]


def test_largest_component_is_noop_for_empty_and_connected_masks():
    empty = np.zeros((4, 4), dtype=bool)
    filtered, removed_components, removed_pixels = lcc.keep_largest_component(empty)
    assert np.array_equal(filtered, empty)
    assert (removed_components, removed_pixels) == (0, 0)

    connected = empty.copy()
    connected[1:3, 1:3] = True
    filtered, removed_components, removed_pixels = lcc.keep_largest_component(connected)
    assert np.array_equal(filtered, connected)
    assert (removed_components, removed_pixels) == (0, 0)

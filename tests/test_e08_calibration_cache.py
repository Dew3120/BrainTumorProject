import materialize_e08_calibration_cache as cache
import numpy as np


def test_hard_scores_match_known_overlap():
    probability = np.zeros((1, 4, 4), dtype=np.float32)
    target = np.zeros((1, 4, 4), dtype=bool)
    probability[0, 1:3, 1:3] = 0.9
    target[0, 1:3, 2:4] = True
    dice, iou = cache.hard_scores(probability, target)
    assert dice[0] == 0.5
    assert iou[0] == 1 / 3

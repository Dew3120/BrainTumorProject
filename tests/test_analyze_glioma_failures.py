import math

import analyze_glioma_failures as analysis
import numpy as np


def test_component_summary_uses_eight_connectivity_and_size_filter():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[3, 3] = True  # Diagonal contact joins the 2x2 block under 8-connectivity.
    mask[6, 6] = True
    raw, at_least_four, largest_fraction = analysis.component_summary(mask)
    assert raw == 2
    assert at_least_four == 1
    assert math.isclose(largest_fraction, 5 / 6)


def test_geometry_reports_centroid_bbox_iou_and_edge_contact():
    reference = np.zeros((8, 8), dtype=bool)
    prediction = np.zeros((8, 8), dtype=bool)
    reference[1:4, 1:5] = True
    prediction[2:5, 3:7] = True
    assert analysis.centroid(reference) == (2.5, 2.0)
    assert analysis.bbox(reference) == (1, 1, 5, 4)
    assert math.isclose(analysis.bbox_iou(analysis.bbox(reference), analysis.bbox(prediction)), 4 / 20)
    assert not analysis.touches_edge(reference)
    prediction[0, 7] = True
    assert analysis.touches_edge(prediction)


def test_orientation_requires_one_registered_filename_token():
    assert analysis.orientation("case_gl_ax_t1.jpg") == "axial"
    assert analysis.orientation("case_gl_co_t1.jpg") == "coronal"
    assert analysis.orientation("case_gl_sa_t1.jpg") == "sagittal"
    assert analysis.orientation("case_gl_t1.jpg") == "unknown"

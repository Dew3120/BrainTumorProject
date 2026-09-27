import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from consolidate_stage3_calibration import EXPERIMENTS


def test_stage3_consolidation_requires_all_nine_items():
    assert EXPERIMENTS == (
        "c01_vector_scaling",
        "c02_dirichlet",
        "c03_isotonic",
        "c04_split_temperature",
        "c05_reliability_weighted",
        "c06_per_category",
        "c07_hierarchical",
        "c08_unified_ece",
        "c09_category_diagrams",
    )

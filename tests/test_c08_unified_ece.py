import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c08_unified_ece import recompute_stored_ece


def test_recompute_stored_ece_uses_counts_and_gaps():
    bins = [
        {
            "count": 2,
            "mean_probability": 0.2,
            "observed_frequency": 0.0,
        }
        for _ in range(15)
    ]
    result = recompute_stored_ece(
        {"pixel_count": 30, "pixel_ece": 0.2, "bins": bins}
    )
    assert abs(result["recomputed_ece"] - 0.2) < 1e-12
    assert result["observations"] == 30

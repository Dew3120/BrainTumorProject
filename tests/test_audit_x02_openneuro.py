import numpy as np

from scripts.audit_x02_openneuro import array_sha256, phash_distance


def test_phash_distance_is_hamming_distance() -> None:
    assert phash_distance("0000000000000000", "000000000000000f") == 4


def test_array_hash_depends_on_array_content() -> None:
    first = np.zeros((2, 2), dtype=np.uint8)
    second = first.copy()
    second[0, 0] = 1
    assert array_sha256(first) != array_sha256(second)

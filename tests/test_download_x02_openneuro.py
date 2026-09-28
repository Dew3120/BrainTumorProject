from pathlib import Path

import pytest

from scripts.download_x02_openneuro import parse_pointer


def test_parse_pointer_reads_locked_size_and_hash(tmp_path: Path) -> None:
    pointer = tmp_path / "sample.nii.gz"
    digest = "a" * 64
    pointer.write_text(
        f"../../.git/annex/objects/key/SHA256E-s123--{digest}.nii.gz/"
        f"SHA256E-s123--{digest}.nii.gz\n",
        encoding="utf-8",
    )
    assert parse_pointer(pointer) == (123, digest)


def test_parse_pointer_rejects_non_annex_content(tmp_path: Path) -> None:
    pointer = tmp_path / "sample.nii.gz"
    pointer.write_text("not an annex pointer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid git-annex pointer"):
        parse_pointer(pointer)

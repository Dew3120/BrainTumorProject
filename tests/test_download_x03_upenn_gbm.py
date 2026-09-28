from scripts.download_x03_upenn_gbm import select_assets


def test_select_assets_requires_exact_expert_pairs() -> None:
    assets = []
    for index in range(147):
        subject = f"sub-{index:03d}"
        for suffix in ("ce-gd_T1w", "seg"):
            name = f"{subject}_{suffix}.nii.gz"
            assets.append(
                {
                    "id": index * 2,
                    "name": name,
                    "size": 10,
                    "digest": "sha256:" + "a" * 64,
                    "browser_download_url": f"https://example.invalid/{name}",
                }
            )
    selected = select_assets({"assets": assets})
    assert len(selected) == 294
    assert {row["kind"] for row in selected} == {"image", "mask"}
    assert len({row["subject_id"] for row in selected}) == 147

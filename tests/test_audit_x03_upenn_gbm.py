from scripts.audit_x03_upenn_gbm import patient_id


def test_patient_id_groups_repeat_scans() -> None:
    assert patient_id("sub-036") == "sub-036"
    assert patient_id("sub-036_2") == "sub-036"

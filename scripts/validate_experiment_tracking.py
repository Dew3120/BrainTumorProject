"""Validate the registered experiment-directory contract and frozen references."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REGISTRY = EXPERIMENTS / "registry.json"
REQUIRED_FILES = {"hypothesis.md", "config.json", "result.json", "decision.md"}
VALID_STATUSES = {
    "REGISTERED_NOT_STARTED",
    "AUDIT_COMPLETE",
    "ACCESS_BLOCKED",
    "COMPLETE",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_experiment(directory, registry_item):
    directory = Path(directory)
    missing = sorted(
        name for name in REQUIRED_FILES if not (directory / name).is_file()
    )
    if missing:
        raise ValueError(f"{registry_item['id']} missing files: {missing}")
    artifacts = directory / "artifacts"
    if not artifacts.is_dir():
        raise ValueError(f"{registry_item['id']} lacks artifacts directory")

    config = load_json(directory / "config.json")
    result = load_json(directory / "result.json")
    if config["experiment_id"] != registry_item["id"]:
        raise ValueError(f"{registry_item['id']} config identity mismatch")
    if result["experiment_id"] != registry_item["id"]:
        raise ValueError(f"{registry_item['id']} result identity mismatch")
    if config["hypothesis_ids"] != registry_item["hypothesis_ids"]:
        raise ValueError(f"{registry_item['id']} hypothesis registration mismatch")
    if config["status"] != registry_item["status"]:
        raise ValueError(f"{registry_item['id']} config status mismatch")
    if registry_item["status"] not in VALID_STATUSES:
        raise ValueError(f"{registry_item['id']} has unknown lifecycle status")
    if registry_item["status"] == "REGISTERED_NOT_STARTED":
        if result["status"] != "NOT_RUN":
            raise ValueError(f"{registry_item['id']} has results before start")
        decision = (directory / "decision.md").read_text(encoding="utf-8")
        if "**Decision:** PENDING" not in decision:
            raise ValueError(f"{registry_item['id']} has a premature decision")
    if registry_item["status"] == "AUDIT_COMPLETE":
        if result["status"] != "AUDIT_COMPLETE":
            raise ValueError(f"{registry_item['id']} audit result status mismatch")
        decision = (directory / "decision.md").read_text(encoding="utf-8")
        if "**Audit decision:**" not in decision:
            raise ValueError(f"{registry_item['id']} lacks an audit decision")
    if registry_item["status"] == "ACCESS_BLOCKED":
        if result["status"] != "ACCESS_BLOCKED":
            raise ValueError(f"{registry_item['id']} access-blocked result mismatch")
        decision = (directory / "decision.md").read_text(encoding="utf-8")
        if "**Decision:** `ACCESS_BLOCKED`" not in decision:
            raise ValueError(f"{registry_item['id']} lacks access-blocked decision")
    return {
        "id": registry_item["id"],
        "status": registry_item["status"],
        "config_sha256": sha256(directory / "config.json"),
        "hypothesis_sha256": sha256(directory / "hypothesis.md"),
        "result_sha256": sha256(directory / "result.json"),
        "decision_sha256": sha256(directory / "decision.md"),
    }


def validate_registry(root=ROOT):
    root = Path(root)
    experiments = root / "experiments"
    registry = load_json(experiments / "registry.json")
    preregistration = root / registry["preregistration"]["path"]
    if sha256(preregistration) != registry["preregistration"]["sha256"]:
        raise ValueError("Preregistration SHA256 mismatch")
    amendments = []
    for amendment in registry.get("amendments", []):
        amendment_path = root / amendment["path"]
        actual_sha256 = sha256(amendment_path)
        if actual_sha256 != amendment["sha256"]:
            raise ValueError(
                f"Preregistration amendment SHA256 mismatch: {amendment['path']}"
            )
        amendments.append(
            {"path": amendment["path"], "sha256": actual_sha256}
        )
    baseline = root / registry["baseline"]["path"]
    if sha256(baseline) != registry["baseline"]["sha256"]:
        raise ValueError("Baseline manifest SHA256 mismatch")

    items = registry["experiments"]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment ID")
    registered_directories = {Path(item["path"]).name for item in items}
    actual_directories = {
        path.name
        for path in experiments.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    if actual_directories != registered_directories:
        raise ValueError(
            "Experiment directory registry mismatch; "
            f"registered={sorted(registered_directories)}, "
            f"actual={sorted(actual_directories)}"
        )

    summaries = []
    for item in items:
        expected = root / item["path"]
        if expected.resolve().parent != experiments.resolve():
            raise ValueError(f"Experiment path escapes registry root: {item['path']}")
        summaries.append(validate_experiment(expected, item))
    return {
        "status": "PASS",
        "experiment_count": len(summaries),
        "preregistration_sha256": registry["preregistration"]["sha256"],
        "amendments": amendments,
        "baseline_manifest_sha256": registry["baseline"]["sha256"],
        "experiments": summaries,
    }


if __name__ == "__main__":
    print(json.dumps(validate_registry(ROOT), indent=2))

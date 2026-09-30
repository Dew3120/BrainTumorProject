"""Build the Stage 9 artifact appendix from committed registries and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/appendix.md"


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def main() -> None:
    frozen = read_json("runs/final_stack/manifest.json")
    registry = read_json("experiments/registry.json")
    figures = read_json("docs/figures/figure_manifest.json")
    amendments = sorted((ROOT / "audit").glob("preregistration_amendment_*.md"))

    lines = [
        "# Artifact, governance, and experiment appendix",
        "",
        (
            "This appendix is generated from committed manifests and registries by "
            "[`docs/build_stage9_appendix.py`](build_stage9_appendix.py). It indexes evidence; "
            "it does not define new results."
        ),
        "",
        "## Frozen stack",
        "",
        f"- Package status: `{frozen['status']}`",
        "- Manifest: [`runs/final_stack/manifest.json`](../runs/final_stack/manifest.json)",
        f"- Manifest SHA256: `{sha256(ROOT / 'runs/final_stack/manifest.json')}`",
        "- Freeze commit: `c2566214bb921c7185fc8949b3e0376492e7113e`",
        "- Frozen Git tree: `439b4fca4b47f12b173c24eaa039b1258f9ddf03`",
        f"- Manifest source commit: `{frozen['source_commit']}`",
        "- Stage 9 rule: the frozen tree is read-only and unchanged.",
        "",
        "### Frozen selections",
        "",
        "| Role | Selection |",
        "|---|---|",
    ]
    for key, value in frozen["selection"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "### Frozen thresholds",
            "",
            "| Threshold | Exact value |",
            "|---|---:|",
        ]
    )
    for key, value in frozen["thresholds"].items():
        rendered = "disabled / null" if value is None else f"`{value}`"
        lines.append(f"| `{key}` | {rendered} |")

    lines.extend(
        [
            "",
            "### Frozen artifact inventory",
            "",
            "| Relative path | Role | Bytes | SHA256 |",
            "|---|---|---:|---|",
        ]
    )
    for item in frozen["artifacts"]:
        path = item["path"]
        role = item["role"].replace("|", "\\|")
        lines.append(f"| `{path}` | {role} | {item['size_bytes']:,} | `{item['sha256']}` |")

    lines.extend(
        [
            "",
            "## Amendments",
            "",
            (
                "The amendment sequence is preserved in numerical order. Each file was "
                "committed before the authorized downstream action continued."
            ),
            "",
            "| Amendment | Title | SHA256 |",
            "|---|---|---|",
        ]
    )
    for path in amendments:
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"| [{path.stem}](../{relative}) | {first_heading(path)} | `{sha256(path)}` |")

    lines.extend(
        [
            "",
            "## Experiment index",
            "",
            (
                "The registry can retain umbrella or future experiments as "
                "`REGISTERED_NOT_STARTED`; that status is not represented as completed work."
            ),
            "",
            "| ID | Status | Hypotheses | Package |",
            "|---|---|---|---|",
        ]
    )
    for experiment in registry["experiments"]:
        hypotheses = ", ".join(experiment.get("hypothesis_ids", [])) or "—"
        path = experiment["path"]
        lines.append(
            f"| `{experiment['id']}` | `{experiment['status']}` | {hypotheses} | "
            f"[`{path}`](../{path}/) |"
        )

    lines.extend(
        [
            "",
            "## Principal audit reports",
            "",
            "- [Pre-registration](../audit/preregistration.md)",
            "- [Review log](../audit/review_log.md)",
            "- [Glioma failure findings](../audit/glioma_failure_findings.md)",
            "- [Calibration findings](../audit/confidence_calibration_findings.json)",
            "- [Stage 3 completion](../audit/stage3_completion_report.md)",
            "- [Stage 4 completion](../audit/stage4_completion_report.md)",
            "- [Stage 5 completion](../audit/stage5_completion_report.md)",
            "- [Stage 6 cross-dataset synthesis](../audit/stage6_cross_dataset.md)",
            "- [Stage 7 OOD completion](../audit/stage7_completion_report.md)",
            "- [Stage 8 final freezing](../audit/stage8_final_freezing.md)",
            "- [Stage 9 metric source ledger](stage9_source_ledger.json)",
            "",
            "## Figure index",
            "",
            (
                "Every figure is generated or wrapped at 300 DPI by "
                "[`docs/build_stage9_figures.py`](build_stage9_figures.py). Source hashes are "
                "recorded in [`docs/figures/figure_manifest.json`](figures/figure_manifest.json)."
            ),
            "",
            "| Figure | Description | SHA256 |",
            "|---|---|---|",
        ]
    )
    for item in figures["figures"]:
        path = item["figure"]
        lines.append(
            f"| [`{Path(path).name}`]({Path(path).relative_to('docs').as_posix()}) | "
            f"{item['description']} | `{item['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduction commands",
            "",
            "```powershell",
            "uv run python docs/build_stage9_figures.py",
            "uv run python docs/build_stage9_appendix.py",
            "uv run python docs/verify_stage9_docs.py",
            "uv run python scripts/verify_final_stack.py",
            "uv run ruff check .",
            "$env:PYTHONPATH=((Resolve-Path '.').Path + ';' + (Resolve-Path 'scripts').Path)",
            "uv run pytest -q",
            "```",
            "",
            (
                "Regenerating documentation must not write to `runs/final_stack/`. The final "
                "verification checks its Git tree and manifest hash."
            ),
            "",
            "## Claim scope",
            "",
            (
                "All metrics are retrospective research results. BRISC values are internal "
                "development-validation results with patient independence unverified. H8 was "
                "rejected on the evaluated independent cohorts. The OOD benchmark is "
                "registered but source-specific. Nothing in this appendix establishes "
                "clinical validity."
            ),
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "experiments": len(registry["experiments"]), "amendments": len(amendments), "artifacts": len(frozen["artifacts"]), "figures": len(figures["figures"])}, indent=2))


if __name__ == "__main__":
    main()

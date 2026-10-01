"""Pinned policy status and read-only model evidence for Stage 4."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gyrus.domain.constants import (
    CLASS_NAMES,
    DISCLAIMER,
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
)
from gyrus.domain.mask_editor import MaskEditSession

from .presentation import ResultPresentation
from .theme import COLORS

REASON_TEXT = {
    "AT_OR_ABOVE_REVIEW_THRESHOLD": "Input differs from development data.",
    "BELOW_REVIEW_THRESHOLD": "OOD review gate passed.",
    "BELOW_THRESHOLD": "Model mask fell below the internal reliability threshold.",
    "RELIABILITY_UNAVAILABLE": "Mask reliability could not be computed.",
    "RELIABILITY_INVALID": "Mask reliability output was invalid.",
    "SEGMENTATION_RELIABILITY_NOT_APPLICABLE": (
        "No segmentation-reliability score applies to this model-predicted label."
    ),
}


class LoadingPanel(QWidget):
    """Name pending work without claiming progress the worker has not reported."""

    STAGES = (
        "Input validated",
        "Frozen preprocessing",
        "Silent model evidence",
        "OOD components and policy",
        "E08 ensemble and C03",
        "R02 reliability and result assembly",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(20, 20, 20, 20)
        column.setSpacing(14)
        title = QLabel("ANALYSIS IN PROGRESS", self)
        title.setObjectName("SectionHeading")
        column.addWidget(title)
        self.lines: list[QLabel] = []
        for name in self.STAGES:
            line = QLabel(f"·  {name}", self)
            line.setObjectName("Secondary")
            column.addWidget(line)
            self.lines.append(line)
        self.activity = QProgressBar(self)
        self.activity.setRange(0, 0)
        self.activity.setTextVisible(False)
        self.activity.setAccessibleName("Frozen analysis active; completion time unknown")
        column.addWidget(self.activity)
        explanation = QLabel(
            "Stages update when the verified worker returns. No intermediate model output is shown.",
            self,
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("Secondary")
        column.addWidget(explanation)
        column.addStretch(1)

    def start(self) -> None:
        for index, label in enumerate(self.lines):
            label.setText(("✓" if index == 0 else "·") + "  " + self.STAGES[index])
        self.lines[1].setText("◌  Frozen preprocessing and evidence in progress")

    def complete(self) -> None:
        for index, label in enumerate(self.lines):
            label.setText(f"✓  {self.STAGES[index]}")


class ResultInspector(QWidget):
    """Status stays visible while provenance and model details scroll."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 16, 16, 16)
        column.setSpacing(12)
        self.status_card = QFrame(self)
        self.status_card.setObjectName("StatusCard")
        card_column = QVBoxLayout(self.status_card)
        self.status = QLabel("", self.status_card)
        self.status.setObjectName("ResultStatus")
        card_column.addWidget(self.status)
        self.reason = QLabel("", self.status_card)
        self.reason.setWordWrap(True)
        card_column.addWidget(self.reason)
        self.edited_badge = QLabel("USER EDITED", self.status_card)
        self.edited_badge.setObjectName("UserEditedBadge")
        self.edited_badge.setVisible(False)
        card_column.addWidget(self.edited_badge)
        column.addWidget(self.status_card)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget(scroll)
        rows = QVBoxLayout(body)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(12)
        self.ood = self._section(rows, "OOD GATE")
        self.category = self._section(rows, "MODEL-PREDICTED CATEGORY")
        self.probabilities = self._section(rows, "RAW CLASS PROBABILITIES")
        self.reliability = self._section(rows, "INTERNAL MASK RELIABILITY")
        self.measurements = self._section(rows, "MODEL MASK · PIXELS ONLY")
        self.caveat = self._section(rows, "INTERPRETATION")
        self.provenance_toggle = QToolButton(body)
        self.provenance_toggle.setText("Provenance ▸")
        self.provenance_toggle.setCheckable(True)
        self.provenance_toggle.toggled.connect(self._toggle_provenance)
        rows.addWidget(self.provenance_toggle)
        self.provenance = QPlainTextEdit(body)
        self.provenance.setReadOnly(True)
        self.provenance.setObjectName("Provenance")
        self.provenance.setVisible(False)
        rows.addWidget(self.provenance)
        rows.addStretch(1)
        scroll.setWidget(body)
        column.addWidget(scroll, 1)
        self.new_case_button = QPushButton("Choose another image", self)
        self.edit_mask_button = QPushButton("Edit mask", self)
        self.save_history_button = QPushButton("Save to local history", self)
        self.edit_mask_button.setAccessibleName("Edit a separate copy of the model mask")
        column.addWidget(self.edit_mask_button)
        column.addWidget(self.save_history_button)
        column.addWidget(self.new_case_button)

    def _section(self, layout: QVBoxLayout, title: str) -> QLabel:
        heading = QLabel(title, self)
        heading.setObjectName("Metadata")
        layout.addWidget(heading)
        value = QLabel("", self)
        value.setWordWrap(True)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(value)
        return value

    def _toggle_provenance(self, checked: bool) -> None:
        self.provenance_toggle.setText("Provenance ▾" if checked else "Provenance ▸")
        self.provenance.setVisible(checked)

    def set_result(self, presentation: ResultPresentation) -> None:
        result = presentation.result
        status = result["final_policy"]["status"]
        review = status == "REVIEW_REQUIRED"
        self.status.setText(("⚠ " if review else "✓ ") + status)
        self.edited_badge.setVisible(False)
        self.edit_mask_button.setEnabled(presentation.mask is not None)
        self.status_card.setProperty("disposition", "review" if review else "standard")
        self.status_card.style().unpolish(self.status_card)
        self.status_card.style().polish(self.status_card)
        reasons = result["final_policy"]["reasons"]
        self.reason.setText(
            "Both internal review gates passed."
            if not review
            else "Review reason: " + " ".join(REASON_TEXT.get(code, code) for code in reasons)
        )
        ood = result["ood"]
        self.ood.setText(
            f"{ood['status']} · score {ood['score']:.6f}\n"
            f"Review threshold {OOD_REVIEW_THRESHOLD_TEXT}"
        )
        classification = result["classification"]
        category = classification["predicted_category"]
        self.category.setText(f"Model-predicted label: {category}")
        self.probabilities.setText(
            "\n".join(
                f"{name.replace('_', ' ')}  {value:.4f}"
                for name, value in zip(CLASS_NAMES, classification["probabilities"], strict=True)
            )
        )
        reliability = result["reliability"]
        score = reliability["score"]
        self.reliability.setText(
            ("Not applicable to a no-tumor prediction" if score is None else f"{score:.6f}")
            + f"\nStandard threshold {RELIABILITY_THRESHOLD_TEXT}"
        )
        spatial = result["spatial"]
        if spatial is None:
            self.measurements.setText("No model mask disclosed. Classification-only output.")
        else:
            stats = spatial["statistics"]
            self.measurements.setText(
                f"Area {stats['foreground_pixels']} / {stats['total_pixels']} model pixels\n"
                f"Components {stats['component_count']} · box {stats['bounding_box_xywh']}"
            )
        caveat = (
            "This is an internal research policy, not a clinical assessment. "
            "In evaluated external cohorts, the frozen OOD policy sent every case "
            "to review while segmentation transferred poorly. External reliability "
            "was miscalibrated, and OOD is weak at ranking whether an individual "
            "mask is correct. E08 combines three model outputs, adding complexity."
        )
        if category == "glioma":
            caveat += " Glioma reliability ranking was weaker in internal development tests."
        self.caveat.setText(caveat)
        self.provenance.setPlainText(
            json.dumps(
                {
                    "case_id": result["case_id"],
                    "job_id": presentation.job_id,
                    "input_sha256": result["input"]["source_sha256"],
                    "manifest_sha256": result["stack_manifest_sha256"],
                    "device": result["execution"]["device"],
                    "duration_ms": result["execution"]["duration_ms"],
                    "segmentation_threshold": spatial["segmentation_threshold"] if spatial else None,
                    "ood_reason": ood["reason"],
                    "policy_reasons": reasons,
                    "e08_route": presentation.route,
                    "raw_probability_sha256": presentation.raw_probability_sha256,
                    "stack": "classifier v1 · E08 · C03 · R02 · O05",
                },
                indent=2,
            )
        )

    def set_edit_status(self, session: MaskEditSession) -> None:
        """Keep frozen result metrics separate from an explicitly saved human layer."""
        saved = session.saved_mask is not None and session.saved_provenance is not None
        self.edited_badge.setVisible(saved)
        score = session.reliability_score
        if score is not None:
            self.reliability.setText(
                f"{score:.6f}\nStandard threshold {RELIABILITY_THRESHOLD_TEXT}\n"
                "Applies to original model mask only; never recalculated on manual edits."
            )
        if saved:
            self.measurements.setText(
                self.measurements.text().split("\nHuman-edited layer:")[0]
                + f"\nHuman-edited layer: {int(session.saved_mask.sum())} pixels"
            )
            provenance = json.loads(self.provenance.toPlainText())
            provenance["human_edit"] = session.saved_provenance.to_dict()
            self.provenance.setPlainText(json.dumps(provenance, indent=2))


class OodReviewDialog(QDialog):
    """Amber disclosure gate; closing it chooses another image."""

    def __init__(self, score: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gyrus — input review")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setObjectName("OodReviewDialog")
        column = QVBoxLayout(self)
        column.setSpacing(14)
        heading = QLabel("⚠  INPUT OUTSIDE REGISTERED DEVELOPMENT DISTRIBUTION", self)
        heading.setObjectName("ReviewHeading")
        column.addWidget(heading)
        for text in (
            f"OOD score: {score:.6f}",
            f"Review threshold: {OOD_REVIEW_THRESHOLD_TEXT}",
            "OOD policy: REVIEW_REQUIRED · REFUSE disabled",
            (
                "This source differs from the BRISC development distribution. The output "
                "remains available for research inspection, but requires review. The OOD "
                "score is weak at telling whether this individual mask is correct. On "
                "evaluated external cohorts, this policy placed every case under review."
            ),
        ):
            line = QLabel(text, self)
            line.setWordWrap(True)
            line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            column.addWidget(line)
        actions = QHBoxLayout()
        self.choose_another = QPushButton("Choose another image", self)
        self.view_output = QPushButton("View research output", self)
        self.view_output.setObjectName("PrimaryAction")
        self.choose_another.clicked.connect(self.reject)
        self.view_output.clicked.connect(self.accept)
        actions.addWidget(self.choose_another)
        actions.addStretch(1)
        actions.addWidget(self.view_output)
        column.addLayout(actions)
        disclaimer = QLabel(DISCLAIMER, self)
        disclaimer.setObjectName("Metadata")
        column.addWidget(disclaimer)
        self.setStyleSheet(
            f"QDialog#OodReviewDialog {{ background: {COLORS['surface']}; "
            f"border: 2px solid {COLORS['review']}; }}"
        )

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
from gyrus.domain.explanation import (
    ood_explanation,
    policy_explanation,
    reliability_explanation,
)
from gyrus.domain.mask_editor import MaskEditSession

from .presentation import ResultPresentation
from .theme import COLORS


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
        self.category = self._section(rows, "MODEL-PREDICTED LABEL")
        self.category.setObjectName("CategoryValue")
        probability_heading = QLabel("CATEGORY PROBABILITIES", body)
        probability_heading.setObjectName("Metadata")
        rows.addWidget(probability_heading)
        self.probabilities = QLabel("", body)
        self.probabilities.setVisible(False)
        self.probability_rows: list[tuple[QLabel, QProgressBar]] = []
        for name in CLASS_NAMES:
            line = QHBoxLayout()
            label = QLabel(name.replace("_", " ").title(), body)
            label.setFixedWidth(92)
            bar = QProgressBar(body)
            bar.setObjectName("ClassProbabilityBar")
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(9)
            value = QLabel("0.00%", body)
            value.setFixedWidth(64)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            line.addWidget(label)
            line.addWidget(bar, 1)
            line.addWidget(value)
            rows.addLayout(line)
            self.probability_rows.append((value, bar))
        self.measurements = self._section(rows, "HIGHLIGHTED REGION · MODEL PIXELS")
        self.ood = self._section(rows, "INPUT SIMILARITY CHECK")
        self.reliability = self._section(rows, "MASK RELIABILITY · INTERNAL ESTIMATE")
        self.caveat = self._section(rows, "WHAT THIS MEANS")
        self.provenance_toggle = QToolButton(body)
        self.provenance_toggle.setText("Technical details ▸")
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
        self.export_pdf_button = QPushButton("Export PDF", self)
        self.edit_mask_button.setAccessibleName("Edit a separate copy of the model mask")
        column.addWidget(self.edit_mask_button)
        column.addWidget(self.save_history_button)
        column.addWidget(self.export_pdf_button)
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
        self.provenance_toggle.setText("Technical details ▾" if checked else "Technical details ▸")
        self.provenance.setVisible(checked)

    def set_result(self, presentation: ResultPresentation) -> None:
        result = presentation.result
        status = result["final_policy"]["status"]
        review = status == "REVIEW_REQUIRED"
        self.status.setText("⚠  REVIEW REQUIRED" if review else "✓  STANDARD")
        self.edited_badge.setVisible(False)
        self.edit_mask_button.setEnabled(presentation.mask is not None)
        self.status_card.setProperty("disposition", "review" if review else "standard")
        self.status_card.style().unpolish(self.status_card)
        self.status_card.style().polish(self.status_card)
        reasons = result["final_policy"]["reasons"]
        self.reason.setText(policy_explanation(result))
        ood = result["ood"]
        self.ood.setText(
            f"{ood_explanation(result)}\n"
            f"Score {ood['score']:.4f} · review from {OOD_REVIEW_THRESHOLD_TEXT}"
        )
        classification = result["classification"]
        category = classification["predicted_category"]
        self.category.setText(category.replace("_", " ").title())
        self.probabilities.setText(
            "\n".join(
                f"{name.replace('_', ' ')}  {value:.4f}"
                for name, value in zip(CLASS_NAMES, classification["probabilities"], strict=True)
            )
        )
        for (value_label, bar), probability in zip(
            self.probability_rows, classification["probabilities"], strict=True
        ):
            bar.setValue(round(probability * 1000))
            value_label.setText(f"{probability * 100:.2f}%")
        reliability = result["reliability"]
        score = reliability["score"]
        if score is None:
            self.reliability.setText(reliability_explanation(result))
        else:
            self.reliability.setText(
                f"Score {score:.4f} · standard from {RELIABILITY_THRESHOLD_TEXT}\n"
                + reliability_explanation(result)
            )
        spatial = result["spatial"]
        if spatial is None:
            self.measurements.setText("No model mask disclosed. Classification-only output.")
        else:
            stats = spatial["statistics"]
            box = stats["bounding_box_xywh"]
            box_text = "No bounding box" if box is None else f"{box[2]} × {box[3]} px bounding box"
            self.measurements.setText(
                f"{stats['foreground_pixels']:,} highlighted pixels · {box_text}\n"
                f"{stats['component_count']} separate region(s). Physical size is unknown."
            )
        caveat = (
            "The coloured region is a model suggestion, not a verified tumor boundary. "
            "On independent external datasets, segmentation performed poorly. "
            "The input-similarity check cannot tell whether this individual mask is correct."
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
                    "ood_score_full_precision": ood["score"],
                    "reliability_score_full_precision": score,
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

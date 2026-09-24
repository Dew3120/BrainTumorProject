"""Load and execute either one segmenter or a registered routed segmenter."""

import csv
import hashlib
import json
from pathlib import Path

import segmentation_overfit_gate as overfit
import torch


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


class RoutedSegmenter:
    """Route each case to a specialist or fallback model using frozen paths."""

    def __init__(self, specialist, fallback, specialist_paths):
        self.specialist = specialist
        self.fallback = fallback
        self.specialist_paths = {normalized_path(path) for path in specialist_paths}

    def eval(self):
        self.specialist.eval()
        self.fallback.eval()
        return self

    def logits_for_rows(self, images, rows):
        if len(images) != len(rows):
            raise ValueError("Routed batch image/row length mismatch")
        use_specialist = torch.tensor(
            [normalized_path(row["path"]) in self.specialist_paths for row in rows],
            dtype=torch.bool,
            device=images.device,
        )
        output = torch.empty(
            (len(images), 1, images.shape[-2], images.shape[-1]),
            dtype=images.dtype,
            device=images.device,
        )
        if use_specialist.any():
            output[use_specialist] = self.specialist(images[use_specialist])
        if (~use_specialist).any():
            output[~use_specialist] = self.fallback(images[~use_specialist])
        return output


def logits_for_rows(model, images, rows):
    if isinstance(model, RoutedSegmenter):
        return model.logits_for_rows(images, rows)
    return model(images)


def load_selected_system(run, checkpoint, device, expected_checkpoint_sha256=None):
    """Load a selected checkpoint and optional fail-closed routing specification."""
    run = Path(run).resolve()
    checkpoint = Path(checkpoint).resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8-sig"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8-sig"))
    selected = result["top3"][0]
    if checkpoint.name != selected["checkpoint"]:
        raise ValueError("Checkpoint is not the selected rank-1 checkpoint")
    checkpoint_hash = sha256(checkpoint)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_hash != expected_checkpoint_sha256.lower()
    ):
        raise ValueError("Selected checkpoint SHA256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != result["best_epoch"] or state["config"] != config:
        raise ValueError("Checkpoint metadata differs from the selected run")
    primary = overfit.make_diagnostic_model("linear", "compact").to(device)
    primary.load_state_dict(state["model_state"])

    routing_path = run / "routing.json"
    if not routing_path.exists():
        return primary
    routing = json.loads(routing_path.read_text(encoding="utf-8-sig"))
    if routing.get("kind") != "classifier_predicted_glioma_specialist":
        raise ValueError("Unsupported segmentation routing specification")
    if routing["specialist_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Routing specialist checkpoint hash mismatch")
    assignments_path = run / routing["assignments_file"]
    if sha256(assignments_path) != routing["assignments_sha256"]:
        raise ValueError("Routing assignment hash mismatch")
    assignments = read_csv(assignments_path)
    if len(assignments) != 680:
        raise ValueError("Routing assignment population mismatch")
    assignment_paths = [normalized_path(row["path"]) for row in assignments]
    if len(set(assignment_paths)) != len(assignment_paths):
        raise ValueError("Duplicate routing assignment path")
    specialist_paths = {
        row["path"] for row in assignments if row["selected_segmenter"] == "specialist"
    }
    allowed = {"specialist", "general_v1"}
    if {row["selected_segmenter"] for row in assignments} - allowed:
        raise ValueError("Unknown selected segmenter in routing assignment")

    fallback_checkpoint = Path(routing["fallback_checkpoint"]).resolve()
    fallback_hash = sha256(fallback_checkpoint)
    if fallback_hash != routing["fallback_checkpoint_sha256"]:
        raise ValueError("Routing fallback checkpoint hash mismatch")
    fallback_state = torch.load(
        fallback_checkpoint, map_location="cpu", weights_only=True
    )
    fallback = overfit.make_diagnostic_model("linear", "compact").to(device)
    fallback.load_state_dict(fallback_state["model_state"])
    routed = RoutedSegmenter(primary, fallback, specialist_paths)
    routed.eval()
    return routed

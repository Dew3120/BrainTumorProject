"""Validate a selected image without blocking Qt's event loop."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, Signal

from gyrus.domain.errors import InputValidationError
from gyrus.domain.input_contract import (
    InputDescriptor,
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)


@dataclass(frozen=True, slots=True)
class ValidatedSelection:
    descriptor: InputDescriptor
    preview_png: bytes
    model_gray_128: bytes


class InputValidationThread(QThread):
    """Return immutable metadata and display pixels; inference still revalidates."""

    validated = Signal(object)
    rejected = Signal(str, str)

    def __init__(self, image_path: str | Path) -> None:
        super().__init__()
        self.image_path = str(image_path)

    def run(self) -> None:
        try:
            decoded = validate_and_decode_image(self.image_path)
            processed = preprocess_for_frozen_stack(decoded)
            image = Image.fromarray(decoded.gray_uint8)
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            model_gray = (
                (processed.segmenter_chw[0] * 255.0).round().clip(0, 255)
            ).astype("uint8")
            self.validated.emit(
                ValidatedSelection(
                    descriptor=decoded.descriptor,
                    preview_png=buffer.getvalue(),
                    model_gray_128=model_gray.tobytes(),
                )
            )
        except InputValidationError as exc:
            self.rejected.emit(exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - file boundary must fail closed
            self.rejected.emit("INPUT_PREVIEW_FAILED", type(exc).__name__)

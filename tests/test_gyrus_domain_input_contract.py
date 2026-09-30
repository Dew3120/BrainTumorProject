from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gyrus.domain.errors import InputValidationError
from gyrus.domain.input_contract import (
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from scripts.final_stack_runtime import preprocess_classifier, preprocess_segmenter


def test_png_decoding_and_preprocessing_match_frozen_runtime(tmp_path: Path):
    pixels = np.arange(32 * 64, dtype=np.uint16).reshape(32, 64) % 256
    path = tmp_path / "case.png"
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)

    decoded = validate_and_decode_image(path)
    prepared = preprocess_for_frozen_stack(decoded)

    assert decoded.descriptor.width == 64
    assert decoded.descriptor.height == 32
    assert prepared.segmenter_chw.shape == (1, 128, 128)
    assert prepared.classifier_chw.shape == (3, 224, 224)
    assert prepared.segmenter_chw.dtype == np.float32
    assert prepared.classifier_chw.dtype == np.float32
    assert not prepared.segmenter_chw.flags.writeable
    np.testing.assert_array_equal(
        prepared.segmenter_chw,
        preprocess_segmenter(decoded.gray_uint8),
    )
    np.testing.assert_array_equal(
        prepared.classifier_chw,
        preprocess_classifier(decoded.gray_uint8),
    )


def test_alpha_image_fails_closed(tmp_path: Path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (32, 32), (10, 20, 30, 128)).save(path)
    with pytest.raises(InputValidationError, match="transparency") as captured:
        validate_and_decode_image(path)
    assert captured.value.code == "INPUT_ALPHA_UNSUPPORTED"


def test_extension_and_decoded_format_must_agree(tmp_path: Path):
    path = tmp_path / "wrong.png"
    Image.new("L", (32, 32), 10).save(path, format="JPEG")
    with pytest.raises(InputValidationError) as captured:
        validate_and_decode_image(path)
    assert captured.value.code == "INPUT_FORMAT_MISMATCH"


def test_exif_orientation_is_rejected_instead_of_silently_rotated(tmp_path: Path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("L", (32, 48), 10)
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    with pytest.raises(InputValidationError) as captured:
        validate_and_decode_image(path)
    assert captured.value.code == "INPUT_EXIF_ORIENTATION_UNSUPPORTED"

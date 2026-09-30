import ast
from pathlib import Path

from gyrus.domain.constants import (
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)

ROOT = Path(__file__).resolve().parents[1]


def test_domain_layer_has_no_qt_torch_or_network_imports():
    forbidden = ("PySide", "PyQt", "torch", "socket", "requests", "httpx", "urllib")
    violations = []
    for path in sorted((ROOT / "gyrus/domain").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith(forbidden):
                    violations.append(f"{path.name}:{node.lineno}:{name}")
    assert violations == []


def test_registered_threshold_text_is_never_rounded():
    assert SEGMENTATION_THRESHOLD_TEXT == "0.50"
    assert RELIABILITY_THRESHOLD_TEXT == "0.9490250173235835"
    assert OOD_REVIEW_THRESHOLD_TEXT == "0.36854275476100085"

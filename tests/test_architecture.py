"""The line the design depends on: no learned model in the measurement loop.

It is not an anti-ML position -- see :mod:`layoutval.authoring`, where a
segmentation model is genuinely the right tool.  It is that ML belongs where its
failure mode is "an engineer adjusts a box" rather than "a safety telltale defect
ships".  A test is the only thing that keeps that line where it is.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "layoutval"

MEASUREMENT_PATH = [
    "measure.py",
    "verdict.py",
    "pipeline.py",
    "calibration.py",
    "displayfind.py",
    "capture.py",
    "residual.py",
    "repeatability.py",
    "linearity.py",
    "types.py",
    "profile.py",
]

FORBIDDEN_PREFIXES = (
    "torch", "torchvision", "tensorflow", "keras", "ultralytics", "mmdet",
    "detectron2", "segment_anything", "sam2", "transformers", "onnxruntime",
    "openvino", "paddle", "sklearn", "layoutval.authoring",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", MEASUREMENT_PATH)
def test_measurement_path_imports_no_learned_model(module):
    for name in _imports(SRC / module):
        assert not name.startswith(FORBIDDEN_PREFIXES), f"{module} imports {name}"


def test_authoring_is_not_reachable_from_the_measurement_path():
    for module in MEASUREMENT_PATH:
        assert "authoring" not in _imports(SRC / module), module


def test_authoring_declares_its_model_dependency_as_a_protocol_only():
    """The segmentation backend must stay a protocol, so the package never
    depends on a model at import time."""
    text = (SRC / "authoring.py").read_text()
    assert "class MaskProposer(Protocol)" in text
    for name in _imports(SRC / "authoring.py"):
        assert not name.startswith(FORBIDDEN_PREFIXES[:-1]), name


def test_runtime_dependencies_are_all_permissively_licensed():
    """AGPL and non-commercial weights are the trap this domain is full of."""
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text()
    deps = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0].lower()
    for banned in ("ultralytics", "superpoint", "surya", "yolov5", "yolov8"):
        assert banned not in deps, banned

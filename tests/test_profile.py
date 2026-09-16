"""The inventory file: what it holds, and what it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layoutval.profile import LayoutProfile, import_design_tree
from layoutval.types import (
    AngleModel,
    ElementKind,
    ElementSpec,
    PositionModel,
    Tolerance,
)

EXPORT = json.loads((Path(__file__).resolve().parents[1] / "examples" / "design_export.json").read_text())


def test_design_import_accumulates_nested_positions():
    """Every tool this targets stores child positions relative to the parent."""
    profile = import_design_tree(EXPORT, screen="main", display_size=(960, 360))
    assert profile["TELLTALE_ABS"].bbox == (304.0, 74.0, 54.0, 40.0)
    assert profile["NEEDLE_SPEED"].bbox == (676.0, 116.0, 168.0, 168.0)
    assert all(e.source == "design" for e in profile)


def test_design_import_maps_node_types_to_kinds():
    profile = import_design_tree(EXPORT, screen="main", display_size=(960, 360))
    assert profile["TELLTALE_ABS"].kind is ElementKind.TELLTALE
    assert profile["LABEL_SPEED_UNITS"].kind is ElementKind.TEXT
    assert profile["NEEDLE_SPEED"].kind is ElementKind.NEEDLE


def test_design_import_skips_containers():
    profile = import_design_tree(EXPORT, screen="main", display_size=(960, 360))
    assert "telltale_row" not in {e.id for e in profile}
    assert "gauge_group" not in {e.id for e in profile}


def test_design_import_honours_a_field_map():
    tree = {"label": "root", "left": 0, "top": 0, "w": 100, "h": 50, "kids": [
        {"label": "ICON", "left": 5, "top": 6, "w": 10, "h": 10, "kids": []},
    ]}
    profile = import_design_tree(
        tree, screen="s", display_size=(100, 50),
        field_map={"id": "label", "x": "left", "y": "top", "width": "w",
                   "height": "h", "children": "kids"},
    )
    assert profile["ICON"].bbox == (5.0, 6.0, 10.0, 10.0)


def test_profile_round_trips_through_yaml(tmp_path):
    profile = import_design_tree(EXPORT, screen="main", display_size=(960, 360), theme="night")
    needle = profile["NEEDLE_SPEED"]
    needle.kind = ElementKind.NEEDLE
    needle.pivot = (760.0, 200.0)
    needle.angle = AngleModel(kind="linear", angle_at_min=210.0, angle_at_max=-30.0,
                              value_min=0.0, value_max=240.0)
    needle.mask = {"hsv_range": [[0, 120, 80], [10, 255, 255]]}
    needle.estimator = "phase"
    needle.context_px = 11
    needle.occludes = ["GAUGE_PLATE"]
    bar = profile["FUEL_BAR"]
    bar.position = PositionModel(kind="linear", origin=(120.0, 300.0),
                                 direction=(300.0, 0.0), value_min=0.0, value_max=1.0)
    bar.signal = "FUEL_LEVEL"
    profile.reference_values = {"FUEL_LEVEL": 0.6}

    path = tmp_path / "p.yaml"
    profile.save(path, templates={"TELLTALE_ABS": np.full((8, 8, 3), 7, np.uint8)},
                 anchors={"TELLTALE_ABS": (311.0, 74.0)})
    loaded = LayoutProfile.load(path)

    assert loaded.theme == "night"
    assert loaded.reference_values == {"FUEL_LEVEL": 0.6}
    assert loaded["NEEDLE_SPEED"].pivot == (760.0, 200.0)
    assert loaded["NEEDLE_SPEED"].angle.expected_angle(120) == pytest.approx(90.0)
    assert loaded["NEEDLE_SPEED"].mask["hsv_range"][1] == [10, 255, 255]
    assert loaded["NEEDLE_SPEED"].estimator == "phase"
    assert loaded["NEEDLE_SPEED"].context() == (11, 11)
    assert loaded["NEEDLE_SPEED"].occludes == ["GAUGE_PLATE"]
    assert loaded["FUEL_BAR"].position.expected_top_left(0.5) == (270.0, 300.0)
    assert loaded["TELLTALE_ABS"].template_anchor == (311.0, 74.0)
    assert loaded.template("TELLTALE_ABS").shape == (8, 8, 3)


def test_profile_only_records_tolerance_overrides(tmp_path):
    defaults = Tolerance(tol_warn=1.0, tol_fail=2.0)
    profile = LayoutProfile("s", (100, 100), defaults=defaults)
    profile.add(ElementSpec(id="a", bbox=(1, 1, 5, 5), tolerance=defaults))
    profile.add(ElementSpec(id="b", bbox=(9, 9, 5, 5),
                            tolerance=Tolerance(tol_warn=1.0, tol_fail=7.0)))
    path = tmp_path / "p.yaml"
    profile.save(path)
    text = path.read_text()
    assert text.count("tol_fail") == 2  # once in defaults, once as b's override
    loaded = LayoutProfile.load(path)
    assert loaded["a"].tolerance.tol_fail == 2.0
    assert loaded["b"].tolerance.tol_fail == 7.0


def test_duplicate_ids_are_refused():
    profile = LayoutProfile("s", (100, 100))
    profile.add(ElementSpec(id="a", bbox=(1, 1, 5, 5)))
    with pytest.raises(ValueError):
        profile.add(ElementSpec(id="a", bbox=(2, 2, 5, 5)))


def test_validate_catches_the_things_worth_refusing_to_run_on():
    profile = LayoutProfile("s", (100, 100))
    profile.add(ElementSpec(id="outside", bbox=(90.0, 90.0, 40.0, 40.0)))
    profile.add(ElementSpec(id="degenerate", bbox=(10.0, 10.0, 0.0, 5.0)))
    profile.add(ElementSpec(id="needle", kind=ElementKind.NEEDLE, bbox=(10.0, 10.0, 5.0, 5.0)))
    profile.add(ElementSpec(id="narrow", bbox=(10.0, 10.0, 5.0, 5.0),
                            search_margin_px=1.0, tolerance=Tolerance(tol_fail=2.5)))
    profile.add(ElementSpec(id="ghost", bbox=(10.0, 10.0, 5.0, 5.0), occludes=["nope"]))
    profile.add(ElementSpec(
        id="mover", bbox=(10.0, 10.0, 5.0, 5.0),
        position=PositionModel(kind="linear", origin=(10.0, 10.0), direction=(30.0, 0.0)),
    ))
    problems = "\n".join(profile.validate())
    assert "outside" in problems
    assert "degenerate" in problems
    assert "needle element needs a pivot" in problems
    assert "search margin" in problems
    assert "unknown element" in problems
    assert "no stored template" in problems  # a mover cannot use the reference


def test_validate_flags_a_tolerance_the_rig_cannot_honour():
    profile = LayoutProfile("s", (100, 100))
    profile.add(ElementSpec(
        id="tight", bbox=(10.0, 10.0, 5.0, 5.0),
        tolerance=Tolerance(tol_fail=0.2, sigma_px=0.5, spec_tolerance_px=1.0),
    ))
    assert any("defensible floor" in p for p in profile.validate())


def test_reference_requires_a_path():
    with pytest.raises(RuntimeError, match="no reference"):
        LayoutProfile("s", (10, 10)).reference()

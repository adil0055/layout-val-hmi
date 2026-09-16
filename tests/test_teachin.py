"""Differential teach-in: the label is correct by construction, or it is nothing."""

from __future__ import annotations

import numpy as np
import pytest

from layoutval.teachin import TeachInSession, teach_profile
from layoutval.types import ElementKind

from conftest import NOMINAL


def make_session(bench, **kw) -> TeachInSession:
    return TeachInSession(capture_display=bench.grab, driver=bench.rig, settle_s=0.0, **kw)


def test_teach_labels_by_construction(bench):
    """No annotation pass, so no labelling errors and no drift between what the
    tool calls an element and what the CAN matrix calls it."""
    session = make_session(bench)
    spec, template = session.teach("TELLTALE_BATTERY_LOW", kind=ElementKind.TELLTALE)
    assert spec.id == "TELLTALE_BATTERY_LOW"
    assert spec.signal == "TELLTALE_BATTERY_LOW"
    assert spec.source == "teachin"
    design = bench.display.LAYOUT["TELLTALE_BATTERY_LOW"]
    assert abs(spec.bbox[0] - design[0]) < 6
    assert abs(spec.bbox[1] - design[1]) < 6
    assert template.shape[0] > 10 and template.shape[1] > 10


def test_teach_finds_the_right_element_despite_a_brighter_neighbour(bench):
    """PWM banding on a large bright element can out-differ a small telltale.

    The floor has to be measured from the screen against itself, not guessed --
    a fixed threshold here silently returns the wrong region, correctly labelled.
    """
    session = make_session(bench)
    bench.display.state["FUEL_LEVEL"] = 0.6  # large, bright, and not the target
    for signal in ("TELLTALE_OIL_PRESSURE", "TELLTALE_ABS"):
        spec, _ = session.teach(signal, kind=ElementKind.TELLTALE)
        design = bench.display.LAYOUT[signal]
        assert abs(spec.bbox[0] - design[0]) < 10, (signal, spec.bbox, design)
        assert abs(spec.bbox[1] - design[1]) < 10, (signal, spec.bbox, design)
        bench.display.state.update(NOMINAL)


def test_measured_floor_is_above_the_rig_noise(bench):
    session = make_session(bench)
    a, b = bench.grab(), bench.grab()
    floor = session.measured_floor(a, b)
    assert floor > 0
    # Nothing changed, so nothing may survive the threshold it implies.
    mask = session.difference_mask(a, b, floor=floor)
    assert mask.sum() / 255 < 0.001 * mask.size


def test_teach_raises_when_the_signal_did_nothing(bench):
    session = make_session(bench)
    with pytest.raises(RuntimeError, match="measured floor"):
        session.teach("TELLTALE_ABS", on_value=True, off_value=True)


def test_teach_moving_learns_a_travel_model(bench):
    session = make_session(bench)
    spec, template = session.teach_moving(
        "FUEL_LEVEL", [0.0, 0.25, 0.5, 0.75, 1.0], element_id="FUEL_BAR"
    )
    assert spec.position.kind == "linear"
    assert spec.position.direction[0] == pytest.approx(300.0, abs=3.0)
    assert abs(spec.position.direction[1]) < 2.0
    assert spec.position.origin[0] == pytest.approx(120.0, abs=3.0)
    assert "residual" in spec.notes
    assert template.size > 0


def test_teach_moving_needs_enough_sweep_points(bench):
    session = make_session(bench)
    with pytest.raises(ValueError, match="three sweep points"):
        session.teach_moving("FUEL_LEVEL", [0.0, 1.0])


def test_split_static_and_moving_separates_the_needle_from_its_plate(bench):
    """The two need different validation, so teach-in has to tell them apart."""
    session = make_session(bench)
    moving, static = session.split_static_and_moving("SPEED", [0.0, 60.0, 120.0, 180.0, 240.0])
    gauge = (660, 100, 200, 200)
    x, y, w, h = gauge
    assert moving[y : y + h, x : x + w].sum() > 0
    assert static[y : y + h, x : x + w].sum() > 0
    # The needle sweeps the upper arc; the plate ring below it does not move.
    assert np.count_nonzero(static) > np.count_nonzero(moving) * 0.2


def test_teach_profile_collects_failures_instead_of_losing_the_run(bench):
    session = make_session(bench)
    profile, templates = teach_profile(
        session,
        ["TELLTALE_BATTERY_LOW", {"signal": "NOT_A_SIGNAL"}],
        screen="main",
        display_size=bench.display.size,
    )
    assert [e.id for e in profile] == ["TELLTALE_BATTERY_LOW"]
    assert "TELLTALE_BATTERY_LOW" in templates
    assert profile.metadata["teach_failures"]

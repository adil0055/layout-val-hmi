"""Tolerances have to be defensible, and the arithmetic has to be visible."""

from __future__ import annotations

import pytest

from layoutval.types import AngleModel, ElementSpec, PositionModel, Tolerance, resolve_value


def test_no_complaint_without_a_study():
    assert Tolerance(tol_fail=0.1).check_against_sigma() is None


def test_floor_includes_noise_bias_and_spec():
    tol = Tolerance(tol_fail=2.0, sigma_px=0.1, subpixel_bias_px=0.3, spec_tolerance_px=1.0)
    assert tol.defensible_floor() == pytest.approx(0.3 + 0.3 + 1.0)
    assert tol.check_against_sigma() is None


def test_complains_when_the_rig_cannot_honour_the_tolerance():
    tol = Tolerance(tol_fail=1.0, sigma_px=0.4, subpixel_bias_px=0.3, spec_tolerance_px=1.0)
    complaint = tol.check_against_sigma()
    assert complaint is not None
    assert "below the defensible floor" in complaint
    assert "2.50" in complaint


def test_subpixel_bias_is_part_of_the_floor():
    """The term a static-screen study cannot see must still be in the sum."""
    quiet = Tolerance(tol_fail=0.5, sigma_px=0.01, spec_tolerance_px=0.0)
    assert quiet.check_against_sigma() is None
    noisy = Tolerance(tol_fail=0.5, sigma_px=0.01, subpixel_bias_px=1.2)
    assert noisy.check_against_sigma() is not None


def test_linear_position_model():
    p = PositionModel(kind="linear", origin=(10.0, 5.0), direction=(100.0, -20.0),
                      value_min=0.0, value_max=1.0)
    assert p.expected_top_left(0.0) == (10.0, 5.0)
    assert p.expected_top_left(0.5) == (60.0, -5.0)
    assert p.expected_top_left(2.0) == (110.0, -15.0)  # clamped to the modelled range
    assert p.expected_top_left(None) == (10.0, 5.0)


def test_static_position_ignores_value():
    p = PositionModel(origin=(3.0, 4.0))
    assert p.expected_top_left(0.9) == (3.0, 4.0)


def test_angle_model():
    a = AngleModel(kind="linear", angle_at_min=210.0, angle_at_max=-30.0,
                   value_min=0.0, value_max=240.0)
    assert a.expected_angle(0) == pytest.approx(210.0)
    assert a.expected_angle(120) == pytest.approx(90.0)
    assert a.expected_angle(240) == pytest.approx(-30.0)
    assert a.expected_angle(None) is None


def test_resolve_value_accepts_element_id_or_signal():
    spec = ElementSpec(id="FUEL_BAR", signal="FUEL_LEVEL")
    assert resolve_value(spec, {"FUEL_BAR": 0.3}) == 0.3
    assert resolve_value(spec, {"FUEL_LEVEL": 0.7}) == 0.7
    assert resolve_value(spec, None) is None
    assert resolve_value(spec, {"OTHER": 1.0}) is None


def test_template_anchor_offsets_the_expectation():
    spec = ElementSpec(
        id="e",
        bbox=(100.0, 50.0, 40.0, 20.0),
        position=PositionModel(kind="linear", origin=(100.0, 50.0),
                               direction=(200.0, 0.0), value_min=0.0, value_max=1.0),
        template_anchor=(105.0, 53.0),
    )
    assert spec.anchor_offset() == (5.0, 3.0)
    assert spec.template_origin(0.0) == (105.0, 53.0)
    assert spec.template_origin(0.5) == (205.0, 53.0)


def test_context_is_per_axis():
    wide = ElementSpec(id="e", bbox=(0.0, 0.0, 200.0, 10.0))
    pad_x, pad_y = wide.context()
    assert pad_x > pad_y  # a long thin element needs context along its length
    assert wide.context_px is None
    fixed = ElementSpec(id="e", bbox=(0.0, 0.0, 200.0, 10.0), context_px=7)
    assert fixed.context() == (7, 7)

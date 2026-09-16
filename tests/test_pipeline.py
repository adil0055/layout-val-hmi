"""End-to-end: inject a defect with known ground truth, check what is reported."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from layoutval.linearity import apply_linearity, run_linearity, shift_image
from layoutval.pipeline import Pipeline, PipelineOptions
from layoutval.profile import LayoutProfile, import_design_tree
from layoutval.report import annotate, write_junit, write_report
from layoutval.repeatability import apply_study, gate, study_from_measurements
from layoutval.teachin import TeachInSession
from layoutval.types import AngleModel, ElementKind, Tolerance, Verdict

from conftest import VALUES

EXPORT = json.loads(
    (__import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "design_export.json").read_text()
)


def make_profile(bench, reference, tmp_path) -> LayoutProfile:
    """The demo's inventory: design export for geometry, teach-in for templates."""
    profile = import_design_tree(
        EXPORT,
        screen="main",
        display_size=bench.display.size,
        theme="day",
        defaults=Tolerance(tol_warn=1.5, tol_fail=2.5, identity_min=0.80, spec_tolerance_px=1.0),
    )
    session = TeachInSession(capture_display=bench.grab, driver=bench.rig, settle_s=0.0)
    templates, anchors = {}, {}
    for signal in ("TELLTALE_BATTERY_LOW", "TELLTALE_OIL_PRESSURE", "TELLTALE_ABS"):
        spec, template = session.teach(signal, kind=ElementKind.TELLTALE)
        templates[signal] = template
        anchors[signal] = (spec.bbox[0], spec.bbox[1])
    bench.reset()

    fuel, fuel_template = session.teach_moving(
        "FUEL_LEVEL", [0.0, 0.25, 0.5, 0.75, 1.0], element_id="FUEL_BAR"
    )
    templates["FUEL_BAR"] = fuel_template
    profile["FUEL_BAR"].position = fuel.position
    profile["FUEL_BAR"].bbox = fuel.bbox
    profile["FUEL_BAR"].signal = "FUEL_LEVEL"
    profile["FUEL_BAR"].source = "teachin"
    bench.reset()

    needle = profile["NEEDLE_SPEED"]
    needle.kind = ElementKind.NEEDLE
    needle.pivot = bench.display.GAUGE_PIVOT
    needle.signal = "SPEED"
    needle.angle = AngleModel(kind="linear", angle_at_min=210.0, angle_at_max=-30.0,
                              value_min=0.0, value_max=240.0)
    needle.mask = {"hsv_range": [[0, 120, 80], [10, 255, 255]], "close_kernel": 3}
    needle.tolerance = Tolerance(tol_warn=2.0, tol_fail=3.5, angle_warn_deg=1.0,
                                 angle_fail_deg=2.0, spec_tolerance_px=1.5)
    needle.search_margin_px = 12.0

    profile.reference_path = "reference.png"
    profile.reference_values = dict(VALUES)

    cv2.imwrite(str(tmp_path / "reference.png"), reference)
    profile.save(tmp_path / "layout_profile.yaml", templates=templates, anchors=anchors)
    return LayoutProfile.load(tmp_path / "layout_profile.yaml")


@pytest.fixture(scope="module")
def _built(request):
    from conftest import build_bench

    bench = build_bench()
    bench.reset()
    reference = bench.grab()
    tmp = request.config._tmp_path_factory.mktemp("profile")
    profile = make_profile(bench, reference, tmp)
    return bench, reference, profile, tmp


@pytest.fixture()
def built(_built):
    bench, reference, profile, tmp = _built
    bench.reset()
    return bench, reference, profile, tmp


def make_pipeline(bench, profile) -> Pipeline:
    return Pipeline(bench.calibration, profile,
                    options=PipelineOptions(settle=False, frames_per_measurement=9))


def test_clean_run_passes(built):
    bench, reference, profile, _ = built
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    failures = [r for r in report.results if r.verdict is Verdict.FAIL]
    assert not failures, [(r.element_id, r.reason, r.measurement.abs_delta) for r in failures]
    for r in report.results:
        if r.measurement.abs_delta is not None:
            assert r.measurement.abs_delta < 0.6, (r.element_id, r.measurement.abs_delta)


@pytest.mark.parametrize("dx,dy", [(3.0, 1.25), (-3.0, 0.0), (0.0, 4.0)])
def test_injected_offset_is_measured_and_reported(built, dx, dy):
    bench, reference, profile, _ = built
    bench.display.offsets["TELLTALE_ABS"] = (dx, dy)
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "TELLTALE_ABS")
    m = r.measurement
    assert math.hypot(m.dx - dx, m.dy - dy) < 0.5, (m.dx, m.dy)
    assert r.verdict is Verdict.FAIL
    assert r.reason == "position"
    assert m.zncc > 0.9  # correct content, just misplaced


def test_wrong_content_is_not_reported_as_a_position_failure(built):
    bench, reference, profile, _ = built
    bench.display.swapped.add("TELLTALE_BATTERY_LOW")
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "TELLTALE_BATTERY_LOW")
    assert r.verdict is Verdict.FAIL
    assert r.reason == "wrong_content"


def test_absent_element_is_reported_as_absent(built):
    bench, reference, profile, _ = built
    bench.display.hidden.add("TELLTALE_OIL_PRESSURE")
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "TELLTALE_OIL_PRESSURE")
    assert r.verdict is Verdict.FAIL
    assert r.reason in ("absent", "wrong_content")


def test_needle_angle_error_is_reported_as_angle_not_position(built):
    """A needle is a pivot and an angle, and the two are separate defects."""
    bench, reference, profile, _ = built
    bench.display.needle_angle_offset_deg = 4.0
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "NEEDLE_SPEED")
    assert r.reason == "angle"
    assert r.measurement.d_angle_deg == pytest.approx(4.0, abs=0.6)
    assert r.measurement.abs_delta < 1.0  # the pivot did not move


def test_needle_pivot_error_is_reported_as_position(built):
    bench, reference, profile, _ = built
    bench.display.needle_pivot_offset = (6.0, 0.0)
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "NEEDLE_SPEED")
    assert r.verdict is Verdict.FAIL
    assert r.reason == "position"
    assert r.measurement.dx == pytest.approx(6.0, abs=1.0)


def test_moving_element_tracks_its_signal(built):
    """A fuel bar at a different level is not a defect; it is a different state."""
    bench, reference, profile, _ = built
    pipeline = make_pipeline(bench, profile)
    bench.display.state["FUEL_LEVEL"] = 0.25
    report = pipeline.measure_frame(
        bench.grab(), values={"FUEL_LEVEL": 0.25, "SPEED": 120.0}, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "FUEL_BAR")
    assert r.verdict is Verdict.PASS, (r.reason, r.measurement.abs_delta)

    # ...and telling the pipeline the wrong state does make it a failure.
    report = pipeline.measure_frame(bench.grab(), values=VALUES, reference=reference)
    r = next(x for x in report.results if x.element_id == "FUEL_BAR")
    assert r.verdict is Verdict.FAIL


def test_unmodelled_artefact_is_surfaced_by_the_residual_not_by_an_element(built):
    bench, reference, profile, _ = built
    bench.display.artefacts.append((520, 150, 46, 22))
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    orphans = [f for f in report.residual_findings if not f.overlaps]
    assert orphans, report.residual_findings
    hit = max(orphans, key=lambda f: f.area_px)
    assert abs(hit.bbox[0] - 520) < 8 and abs(hit.bbox[1] - 150) < 8
    # Advisory only: it must not on its own fail an element.
    assert all(r.verdict is not Verdict.FAIL for r in report.results)
    assert report.verdict is Verdict.REVIEW


def test_far_displacement_lands_on_the_search_border(built):
    """Beyond the search window the magnitude is unknown -- that is a failure,
    not a pass and not a clamped value."""
    bench, reference, profile, _ = built
    bench.display.offsets["TELLTALE_ABS"] = (40.0, 0.0)
    report = make_pipeline(bench, profile).measure_frame(
        bench.grab(), values=VALUES, reference=reference
    )
    r = next(x for x in report.results if x.element_id == "TELLTALE_ABS")
    assert r.verdict is Verdict.FAIL
    assert r.reason in ("missing_or_displaced", "wrong_content")


def test_report_round_trips_and_writes_artefacts(built, tmp_path):
    bench, reference, profile, _ = built
    bench.display.offsets["TELLTALE_ABS"] = (3.0, 0.0)
    live = bench.grab()
    report = make_pipeline(bench, profile).measure_frame(
        live, values=VALUES, reference=reference
    )
    written = write_report(report, tmp_path, profile=profile, live_display=live,
                           reference=reference, values=VALUES)
    assert (tmp_path / "report.json").exists()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["verdict"] == "FAIL"
    assert any(e["element_id"] == "TELLTALE_ABS" for e in payload["elements"])
    assert payload["elements"][0]["measurement"]["abs_delta"] is not None
    assert "overlay" in written

    junit = write_junit([report], tmp_path / "junit.xml")
    text = junit.read_text()
    assert "<testsuite" in text and 'name="TELLTALE_ABS"' in text

    overlay = annotate(live, report, profile, values=VALUES)
    assert overlay.shape == live.shape
    assert not np.array_equal(overlay, live)


def test_repeatability_and_linearity_measure_different_things(built):
    """A rig can be very repeatable and still carry sub-pixel bias.

    This is the reason the gate needs both: sigma from a static screen says
    nothing about accuracy against a displacement that is not a whole pixel.
    """
    bench, reference, profile, _ = built
    pipeline = make_pipeline(bench, profile)
    samples = {s.id: [] for s in profile}
    for _ in range(6):
        report = pipeline.measure_frame(bench.grab(), values=VALUES, reference=reference)
        for r in report.results:
            if r.element_id in samples:
                samples[r.element_id].append(r.measurement)
    study = study_from_measurements("main", samples)
    assert study.elements["TELLTALE_ABS"].sigma < 0.5

    linearity = run_linearity(profile, reference, values=VALUES, shifts=(0.0, 0.25, 0.5, 0.75))
    assert set(linearity.elements) == {s.id for s in profile}
    for lin in linearity.elements.values():
        assert lin.n > 0

    apply_study(profile, study)
    apply_linearity(profile, linearity)
    for spec in profile:
        assert spec.tolerance.sigma_px is not None
        assert spec.tolerance.defensible_floor() >= spec.tolerance.spec_tolerance_px
    assert isinstance(gate(profile, study).passed, bool)


def test_shift_image_is_a_real_subpixel_shift():
    """Checked with a centroid, which has an exact answer for a symmetric blob."""
    from layoutval.measure import centroid_from_mask

    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    blob = np.exp(-(((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 12.0**2)))
    base = centroid_from_mask(blob)[0]
    for dx, dy in ((0.5, 0.0), (0.0, -0.25), (0.75, 0.75)):
        out = centroid_from_mask(shift_image(blob, dx, dy))[0]
        assert out[0] - base[0] == pytest.approx(dx, abs=0.05)
        assert out[1] - base[1] == pytest.approx(dy, abs=0.05)

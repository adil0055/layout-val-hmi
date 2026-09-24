"""Reflections off the cover glass.

The scenes are the ones that broke validation before any of this existed:
reflections that move between the reference and the test shot -- which is what
a hand-held phone guarantees -- soft ones and flat-sided ones, and one strong
enough to clip. Each is photographed through the simulator, which adds them in
linear light, where light actually adds.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval import glare
from layoutval.server import CaptureSession, _payload
from layoutval.simulator import ClusterDisplay, Reflection, SimulatedRig, VirtualCamera
from tests.conftest import NOMINAL, PATTERN, PATTERN_ORIGIN, SQUARE_PX, VALUES

WINDOW_A = Reflection((0.35, 0.45), (0.25, 0.35), 80, softness=3)
WINDOW_B = Reflection((0.50, 0.45), (0.25, 0.35), 80, softness=3)
LAMP = Reflection((0.30, 0.35), (0.12, 0.20), 220)


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


def shoot(tmp_path, ref_glare, live_glare, *, deglare=True, fault=None):
    """Calibrate clean, take the reference and the test shot under the given glare."""
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    camera = VirtualCamera(display_size=display.size)
    rig = SimulatedRig(display, camera)
    session = CaptureSession(
        tmp_path, pattern_size=PATTERN, square_px=SQUARE_PX, pattern_origin=PATTERN_ORIGIN,
        display_size=display.size, values=VALUES, deglare=deglare)
    rig.show("checkerboard")
    assert session.handle("calibrate", jpeg(rig.read())).verdict == "OK"
    rig.show("main")
    camera.glare = list(ref_glare)
    session.handle("reference", jpeg(rig.read()))
    if fault:
        display.offsets[fault[0]] = fault[1]
    camera.glare = list(live_glare)
    return session.handle("validate", jpeg(rig.read())).report


def element_at(report, x, y):
    def dist(e):
        ex, ey = (float(v) for v in e["element_id"].split("@")[1].split(","))
        return abs(ex - x) + abs(ey - y)
    return min(report["elements"], key=dist)


# -- the correction itself ---------------------------------------------------


def dark_scene():
    rng = np.random.default_rng(0)
    img = np.full((480, 1280, 3), 14, np.uint8)
    for _ in range(25):
        x, y = int(rng.integers(20, 1200)), int(rng.integers(20, 420))
        cv2.putText(img, "88", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    return img


def lit(img, reflection):
    linear = glare.to_linear(img) + (reflection.image(img.shape[:2]) / 255.0)[..., None] ** 2.2
    return glare.to_encoded(linear)


def test_a_frame_compared_with_itself_is_left_exactly_alone():
    img = dark_scene()
    pair = glare.deglare_pair(img, img)
    assert (pair.reference == img).all() and (pair.live == img).all()
    assert not pair.clipped.any() and not pair.footprint.any()


def test_a_flat_sided_reflection_comes_off_edges_and_all():
    """The median tried first left 50-70 level rims along a window frame's edge."""
    img = dark_scene()
    window = Reflection((0.45, 0.5), (0.3, 0.5), 90, softness=2)
    pair = glare.deglare_pair(img, lit(img, window))
    residual = np.abs(pair.live.astype(int) - img.astype(int)).max(axis=2)
    # Away from the window's edge the subtraction is exact to a few levels,
    # artwork included. On the edge, the estimate is smoothed a little against
    # sensor noise and a thin rim is left: 26 levels at worst, in a band six
    # pixels either side.
    shape = window.image(img.shape[:2]) > 45
    band = cv2.dilate(cv2.morphologyEx(shape.astype(np.uint8), cv2.MORPH_GRADIENT,
                                       np.ones((3, 3), np.uint8)), np.ones((13, 13), np.uint8)) > 0
    assert residual[~band].max() <= 8
    assert residual[band].max() <= 30
    assert pair.footprint.any()
    assert pair.subtracted > 60


def test_a_day_theme_is_handled_the_other_way_up():
    img = 255 - dark_scene()            # dark strokes on a light ground
    assert not glare.bright_on_dark(img)
    blob = Reflection((0.5, 0.5), (0.4, 0.6), 60)
    pair = glare.deglare_pair(img, lit(img, blob))
    strokes = img.max(axis=2) < 100
    # The strokes keep their contrast against the ground next to them.
    assert np.abs(pair.live[strokes].astype(int) - img[strokes].astype(int)).mean() < 6


def test_the_simulator_adds_glare_in_linear_light():
    """A reflection lifts black a long way and white hardly at all."""
    display = ClusterDisplay()
    camera = VirtualCamera(display_size=display.size, noise_sigma=0, pwm_amplitude=0)
    black = np.zeros((display.size[1], display.size[0], 3), np.uint8)
    white = np.full_like(black, 230)
    plain_b, plain_w = camera.shoot(black), camera.shoot(white)
    camera.glare = [Reflection((0.5, 0.5), (0.3, 0.3), 80)]
    lift_b = camera.shoot(black).astype(int) - plain_b
    lift_w = camera.shoot(white).astype(int) - plain_w
    h, w = lift_b.shape[:2]
    # 80 levels on black; on white at 230, about 10.
    assert lift_b[h // 2, w // 2, 0] > 60
    assert lift_w[h // 2, w // 2, 0] < lift_b[h // 2, w // 2, 0] / 5


# -- end to end, through the phone's capture session ---------------------------


def test_a_reflection_that_moved_does_not_fail_a_good_screen(tmp_path):
    assert shoot(tmp_path / "on", [WINDOW_A], [WINDOW_B])["verdict"] == "PASS"
    # And the reason this exists: as photographed, the same pair is not a pass.
    assert shoot(tmp_path / "off", [WINDOW_A], [WINDOW_B], deglare=False)["verdict"] != "PASS"


def test_a_real_fault_is_still_measured_through_a_moving_reflection(tmp_path):
    fault = ("TELLTALE_OIL_PRESSURE", (2.0, 0.0))
    clean = element_at(shoot(tmp_path / "clean", [], [], fault=fault), 216, 74)
    glared = element_at(shoot(tmp_path / "glare", [WINDOW_A], [WINDOW_B], fault=fault), 216, 74)
    assert glared["verdict"] != "PASS"
    assert glared["measurement"]["abs_delta"] == pytest.approx(
        clean["measurement"]["abs_delta"], abs=0.15)


def test_what_a_reflection_clipped_is_review_not_a_verdict(tmp_path):
    report = shoot(tmp_path, [], [LAMP])
    assert report["verdict"] == "REVIEW"
    reasons = {e["element_id"]: (e["verdict"], e["reason"]) for e in report["elements"]}
    assert ("REVIEW", "glare") in reasons.values()
    assert not any(v == "FAIL" for v, _ in reasons.values())
    assert any(f["flag"] == "glare" for f in report["flags"])


def test_a_strong_reflection_on_the_reference_is_never_a_quiet_pass(tmp_path):
    """The same lamp in both shots washed a telltale out of the inventory, and a
    2 px fault in it came back PASS: nothing compared two frames that differed."""
    report = shoot(tmp_path, [LAMP], [LAMP], fault=("TELLTALE_OIL_PRESSURE", (2.0, 0.0)))
    assert report["verdict"] != "PASS"
    assert any(f["flag"] == "reference_glare" for f in report["flags"])


def test_the_phone_shows_findings_not_notes(tmp_path):
    from layoutval.server import CaptureRecord

    rec = CaptureRecord(name="x.jpg", action="validate", when="now", verdict="PASS",
                        report={"flags": [
                            {"flag": "pose_resolved", "severity": "note", "detail": "hand-held"},
                            {"flag": "glare", "severity": "review", "detail": "a reflection"},
                        ], "elements": []})
    assert _payload(rec)["flags"] == ["a reflection"]

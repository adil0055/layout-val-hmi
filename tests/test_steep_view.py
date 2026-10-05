"""Photographs taken from well round the screen: a first pose that missed, and a screen not quite flat.

What six photographs of a laptop's cluster showed, nothing on it changed: from
well above and from the left, matched features offered no start, ECC settled
on the keyboard and every element failed; a blurred one started 8-30 px out and
no mapping explained its loose matches to 1 px. With the pose right, the middle
of the screen was still 3-3.5 px out against its sides, along the way the
camera had moved -- a screen about a millimetre out of flat.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from layoutval import anchor
from layoutval.types import (
    ElementKind,
    ElementResult,
    ElementSpec,
    Measurement,
    PositionModel,
    RunReport,
    Tolerance,
    Verdict,
)
from layoutval.verdict import POSITION
from layoutval.viewpoint import VIEW_ANGLE, allow_for_view, camera_centre, parallax

from test_moving_phone import Bench, jpeg, start

DISPLAY = (1920, 1200)
K = np.array([[1900.0, 0, 1288], [0, 1900.0, 966], [0, 0, 1]])


def _looking_from(C, at=(960.0, 600.0)):
    """display -> camera px for a camera at ``C`` (display px; z its distance) looking at ``at``."""
    C = np.asarray(C, float)
    target = np.array([at[0], at[1], 0.0])
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                      # world -> camera
    t = -R @ C
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def test_the_camera_is_found_where_it_was():
    for C in ([960, 600, -2000], [200, 300, -1500], [1700, 1100, -2600]):
        found = camera_centre(_looking_from(C), K)
        assert np.allclose(found, [C[0], C[1], -C[2]], atol=1.0), (C, found)


def _report(shifts):
    specs, report = [], RunReport(screen="t")
    for i, ((x, y), (dx, dy)) in enumerate(shifts):
        spec = ElementSpec(id=f"e{i}", kind=ElementKind.TEXT, bbox=(x - 20, y - 10, 40, 20),
                           position=PositionModel(origin=(x - 20, y - 10)), tolerance=Tolerance())
        specs.append(spec)
        report.results.append(ElementResult(
            element_id=spec.id, verdict=Verdict.FAIL, reason=POSITION,
            measurement=Measurement(element_id=spec.id, dx=dx, dy=dy), tolerance=Tolerance()))
    return specs, report


def test_a_shift_the_change_of_view_explains_is_review_and_one_across_it_still_fails():
    # The camera moved 0.6 of the screen's width to the left: about 18 degrees.
    H_ref, H_live = _looking_from([960, 500, -2000]), _looking_from([-200, 500, -1800])
    axis = parallax(H_ref, H_live, K, [(960, 600)])[0]
    axis /= np.linalg.norm(axis)
    assert abs(axis[0]) > 0.95                   # a move to the side slides points sideways
    across = np.array([-axis[1], axis[0]])
    specs, report = _report([((960, 600), 3.2 * axis), ((900, 700), 3.2 * across),
                             ((1000, 500), 9.0 * axis)])
    assert allow_for_view(report, specs, None, H_ref, H_live, K, DISPLAY) == 1
    verdicts = [(r.verdict, r.reason) for r in report.results]
    assert verdicts[0] == (Verdict.REVIEW, VIEW_ANGLE)
    assert verdicts[1] == (Verdict.FAIL, POSITION)       # across the view's move
    assert verdicts[2] == (Verdict.FAIL, POSITION)       # more than the view explains
    assert report.flags[-1]["flag"] == "view_moved"


def test_with_the_camera_where_it_was_nothing_is_excused():
    H_ref, H_live = _looking_from([960, 500, -2000]), _looking_from([1000, 520, -2000])
    axis = parallax(H_ref, H_live, K, [(960, 600)])[0]
    specs, report = _report([((960, 600), 3.0 * axis / max(np.linalg.norm(axis), 1e-9))])
    assert allow_for_view(report, specs, None, H_ref, H_live, K, DISPLAY) == 0
    assert report.results[0].verdict is Verdict.FAIL and not report.flags


def test_loose_votes_from_a_rough_start_are_brought_close():
    rng = np.random.default_rng(5)
    src = rng.uniform([80, 60], [1840, 1140], (35, 2))
    truth = np.array([[1.01, 0.02, 14.0], [-0.015, 0.99, -9.0], [1e-5, -8e-6, 1.0]])
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), truth).reshape(-1, 2)
    dst += rng.normal(0, 0.9, dst.shape)          # a blurred photo's loose matches
    dst[:5] += rng.uniform(-25, 25, (5, 2))       # and a few that found something else
    G, agreeing = anchor.rough_correction(src, dst)
    assert agreeing >= 28
    err = cv2.perspectiveTransform(src.reshape(-1, 1, 2), G) - cv2.perspectiveTransform(
        src.reshape(-1, 1, 2), truth)
    # Near enough that each element is then matched precisely: a start 15 px
    # out, loose matches, within 2 px everywhere.
    off = np.linalg.norm(err.reshape(-1, 2), axis=1)
    assert off.max() < 2.0 and np.median(off) < 0.8


def test_a_first_pose_that_missed_the_screen_is_replaced_by_one_the_elements_agree_with():
    bench = Bench()
    session = start(bench)
    bench.move(t=(0.08, 0, 0))
    solve = session._solve_pose

    def missed(*args, **kw):
        # Where features and ECC can leave a steep view: 60 px out.
        est = solve(*args, **kw)
        off = np.array([[1, 0, 60.0], [0, 1, -45.0], [0, 0, 1]])
        return dataclasses.replace(est, warp_camera=off @ est.warp_camera)

    session._solve_pose = missed
    report = session.handle("validate", jpeg(bench.shoot())).report
    assert {"rough": True} in report["metadata"]["pose_steps"]
    assert report["verdict"] == "PASS", [
        (e["element_id"], e["reason"], e["measurement"]["abs_delta"])
        for e in report["elements"] if e["verdict"] != "PASS"]


def test_a_view_too_far_round_to_judge_the_tolerance_says_how_close_to_come():
    specs, report = _report([])
    far = allow_for_view(report, specs, None, _looking_from([960, 500, -2000]),
                         _looking_from([-200, 500, -1800]), K, DISPLAY, tol_fail=2.5)
    assert far == 0
    flag = report.flags[-1]
    assert flag["flag"] == "view_moved" and flag["too_far"] and flag["severity"] == "review"
    assert "within about" in flag["detail"]

    specs, report = _report([])
    allow_for_view(report, specs, None, _looking_from([960, 500, -2000]),
                   _looking_from([1060, 520, -2000]), K, DISPLAY, tol_fail=2.5)
    assert not report.flags


def test_too_far_round_a_shift_within_the_unevenness_is_review_whichever_way_it_points():
    H_ref, H_live = _looking_from([960, 500, -2000]), _looking_from([-200, 500, -1800])
    axis = parallax(H_ref, H_live, K, [(960, 600)])[0]
    axis /= np.linalg.norm(axis)
    across = np.array([-axis[1], axis[0]])
    specs, report = _report([((960, 600), 3.2 * across), ((900, 650), 12.0 * across)])
    allow_for_view(report, specs, None, H_ref, H_live, K, DISPLAY, tol_fail=2.5)
    assert report.results[0].verdict is Verdict.REVIEW
    assert report.results[1].verdict is Verdict.FAIL         # more than any unevenness

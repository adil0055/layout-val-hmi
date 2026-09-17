"""Stage 2 and 3: the geometry that sets the whole accuracy budget."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from layoutval.calibration import (
    Calibration,
    DisplayGeometry,
    DriftTracker,
    calibrate_intrinsics,
    chessboard_display_points,
    homography_from_display_edges,
)
from layoutval.capture import capture, median_stack
from layoutval.measure import phase_shift, zncc_match
from conftest import build_bench


def test_chessboard_display_points_are_the_inner_corners():
    pts = chessboard_display_points((3, 2), 10.0, (5.0, 7.0))
    assert len(pts) == 6
    assert tuple(pts[0]) == (15.0, 17.0)
    assert tuple(pts[-1]) == (35.0, 27.0)


def test_homography_from_display_pattern_is_accurate(bench):
    """Method A: the cluster renders the pattern, so the correspondence is exact."""
    assert bench.geometry.method == "display_pattern"
    assert bench.geometry.residual_px < 0.3


def test_rectified_frame_matches_the_framebuffer(bench):
    """The whole point of stage 3: display space, to a fraction of a pixel."""
    rectified = bench.grab()
    truth = bench.display.render()
    g_true = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
    g_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    (px, py), score, _ = zncc_match(g_rect[60:320, 600:900], g_true[80:300, 620:880])
    assert score > 0.95
    assert abs(600 + px - 620) < 0.25
    assert abs(60 + py - 80) < 0.25


def test_sampling_ratio_is_reported(bench):
    ratio = bench.geometry.sampling_ratio()
    assert 1.4 < ratio < 1.8  # the simulator is deliberately under-sampled


def test_display_to_camera_round_trips(bench):
    pts = np.array([[0.0, 0.0], [959.0, 359.0], [480.0, 180.0]])
    back = bench.geometry.camera_to_display(bench.geometry.display_to_camera(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_homography_from_display_edges_is_a_workable_fallback():
    """Method C re-derives the geometry from content, so it is looser -- but the
    line fits must still land well under a pixel of reprojection residual."""
    bench = build_bench()
    bench.rig.show("white")
    frame = bench.undistort(median_stack(capture(bench.rig, n=5)))
    geometry = homography_from_display_edges(frame, display_size=bench.display.size)
    assert geometry.method == "display_edges"
    assert geometry.residual_px < 1.0
    corners = geometry.display_to_camera(np.array([[0.0, 0.0], [960.0, 360.0]]))
    truth = bench.geometry.display_to_camera(np.array([[0.0, 0.0], [960.0, 360.0]]))
    assert np.max(np.abs(corners - truth)) < 6.0


def test_screen_content_matches_the_chessboard(bench):
    """No pattern on the screen, and the same answer.

    Scored the way it matters: rectify with each mapping and measure where the
    elements land against the true framebuffer. benchmarks/calibration_methods.py
    is the full sweep across pose, focus, sampling and distortion.
    """
    from layoutval.autoprofile import profile_from_reference
    from layoutval.calibration import homography_from_screen_content
    from layoutval.measure import measure_translation

    truth = bench.display.render()
    frame = bench.undistort(median_stack(capture(bench.rig, n=5)))
    geometry = homography_from_screen_content(
        truth, frame, display_size=bench.display.size
    )
    assert geometry.method.startswith("screen_content")
    assert geometry.inliers >= 12

    probe = profile_from_reference(truth, screen="p", display_size=bench.display.size)
    assert len(probe) >= 4

    def element_rms(H):
        rect = DisplayGeometry(H=H, display_size=bench.display.size).rectify(frame)
        errs = [
            math.hypot(m.dx, m.dy)
            for spec in probe
            for m in [measure_translation(truth, rect, spec)]
            if m.dx is not None
        ]
        return math.sqrt(sum(e * e for e in errs) / len(errs))

    from_content = element_rms(geometry.H)
    from_board = element_rms(bench.geometry.H)
    assert from_content < 0.35, from_content
    # Within a fifth of a pixel of the pattern, having drawn no pattern.
    assert abs(from_content - from_board) < 0.2, (from_content, from_board)


def test_screen_content_refuses_a_nonsense_mapping(bench):
    """A homography fitted to bad matches can be arithmetically fine and
    geometrically absurd, and nothing downstream would point back here."""
    from layoutval.calibration import _check_display_quad

    # A perspective term strong enough to send the far edge behind the camera
    # turns the quad into a bowtie.
    folded = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, -0.004, 1.0]])
    with pytest.raises(RuntimeError, match="folds"):
        _check_display_quad(folded, (960, 360), (900, 1600))

    # Mirrored: still convex, still the wrong answer.
    mirrored = np.array([[-1.0, 0, 960.0], [0, 1.0, 0], [0, 0, 1.0]])
    with pytest.raises(RuntimeError, match="mirrors"):
        _check_display_quad(mirrored, (960, 360), (900, 1600))
    tiny = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 1.0]])
    with pytest.raises(RuntimeError, match="times the frame area"):
        _check_display_quad(tiny, (960, 360), (900, 1600))


def test_calibrate_intrinsics_refuses_too_few_views():
    with pytest.raises(RuntimeError, match="usable"):
        calibrate_intrinsics([np.zeros((100, 100, 3), np.uint8)] * 3, (9, 6), min_views=8)


def test_calibration_round_trips(tmp_path, bench):
    path = tmp_path / "rig.json"
    bench.calibration.save(path)
    loaded = Calibration.load(path)
    assert np.allclose(loaded.geometry.H, bench.geometry.H)
    assert loaded.geometry.display_size == bench.geometry.display_size
    assert np.allclose(loaded.intrinsics.K, bench.undistort.intrinsics.K)


def test_drift_is_measured_and_alarmed_not_silently_absorbed():
    """Silent compensation is how a rig somebody knocked keeps going green."""
    bench = build_bench()
    bench.rig.show("main")
    base = bench.undistort(median_stack(capture(bench.rig, n=5)))
    roi = (420, 180, 700, 520)
    tracker = DriftTracker(base, roi, alarm_px=2.0)

    still = tracker.measure(bench.undistort(median_stack(capture(bench.rig, n=5))))
    assert still.converged
    assert still.magnitude_px < 1.0
    assert not still.exceeds

    # Knock the rig: shift the camera frame by a known amount.
    knocked_src = bench.undistort(median_stack(capture(bench.rig, n=5)))
    M = np.float32([[1, 0, 6.0], [0, 1, -3.0]])
    knocked = cv2.warpAffine(knocked_src, M, (knocked_src.shape[1], knocked_src.shape[0]))
    est = tracker.measure(knocked)
    assert est.converged
    assert est.magnitude_px == pytest.approx(math.hypot(6.0, 3.0), abs=1.0)
    assert est.exceeds  # loud, not silent

    corrected = tracker.corrected_homography(bench.geometry, est)
    assert not np.allclose(corrected, bench.geometry.H)
    # The correction has to actually undo the knock in display space.
    before = bench.geometry.rectify(knocked)
    after = bench.geometry.rectify(knocked, corrected)
    truth = bench.geometry.rectify(knocked_src)
    win = (slice(60, 300), slice(600, 900))
    err_before = phase_shift(truth[win], before[win])[0]
    err_after = phase_shift(truth[win], after[win])[0]
    assert math.hypot(*err_after) < math.hypot(*err_before)
    assert math.hypot(*err_after) < 1.0


def test_undistortion_is_applied(bench):
    """Without it, elements near the frame edge read as shifted on a good cluster."""
    bench.rig.show("main")
    raw = median_stack(capture(bench.rig, n=5))
    undistorted = bench.undistort(raw)
    assert undistorted.shape == raw.shape
    assert not np.array_equal(raw, undistorted)

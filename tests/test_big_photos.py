"""A phone's full-resolution photograph: 24 MP on a current iPhone.

At that size two steps took long enough, and one used enough memory, for the
phone to give up on the request -- which on an iPhone reads only "Load failed".
The hand-held pose re-solve took 31-46 s and 3.4 GB when the camera had moved
between shots, and the chessboard search 12-18 s. Both now work on a reduced
copy and hand back full-resolution answers; these check the answers.
"""

from __future__ import annotations

import cv2
import numpy as np

from layoutval.calibration import (
    DriftTracker,
    _orient_corners,
    chessboard_display_points,
    find_chessboard,
)
from layoutval.server import CaptureSession
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera
from tests.conftest import NOMINAL, PATTERN, PATTERN_ORIGIN, SQUARE_PX, VALUES


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


def test_a_pose_solved_small_is_the_same_pose_full_size():
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur(rng.integers(0, 255, (1400, 1800), dtype=np.uint8), (0, 0), 3)
    warp = np.array([[1.0, 0.004, 6.3], [-0.004, 1.0, -2.7], [0.0, 0.0, 1.0]])
    # Both frames cut from inside the canvas, so neither has an empty border
    # for the alignment to lock onto; the warp re-expressed for the cut.
    cut = np.array([[1.0, 0.0, -100.0], [0.0, 1.0, -100.0], [0.0, 0.0, 1.0]])
    true = cut @ warp @ np.linalg.inv(cut)
    ref = big[100:1300, 100:1700]
    live = cv2.warpPerspective(big, warp, (1800, 1400), flags=cv2.INTER_CUBIC)[100:1300, 100:1700]
    size = (800, 600)
    small = DriftTracker(cv2.resize(ref, size, interpolation=cv2.INTER_AREA), (0, 0, *size),
                         motion=cv2.MOTION_HOMOGRAPHY)
    est = small.measure(cv2.resize(live, size, interpolation=cv2.INTER_AREA)).rescaled(0.5)
    probe = np.array([[[400.0, 300.0]], [[1200.0, 900.0]], [[800.0, 600.0]]])
    got = cv2.perspectiveTransform(probe, est.warp_camera)
    want = cv2.perspectiveTransform(probe, true)
    assert float(np.abs(got - want).max()) < 0.05


def test_a_chessboard_in_a_24_mp_photo_is_found_small_and_refined_full_size():
    display = ClusterDisplay()
    cam = VirtualCamera(display_size=display.size, sensor_size=(5712, 4284),
                        sampling_ratio=4.5, k1=0.0, k2=0.0, tilt_deg=8.0, roll_deg=-5.0)
    rig = SimulatedRig(display, cam)
    rig.show("checkerboard")
    found = find_chessboard(rig.read(), PATTERN)
    assert found is not None
    corners, refined = found
    assert refined
    truth = cv2.perspectiveTransform(
        chessboard_display_points(PATTERN, SQUARE_PX, PATTERN_ORIGIN).reshape(-1, 1, 2),
        cam.H_true).reshape(-1, 2)
    err = np.linalg.norm(_orient_corners(corners, PATTERN) - truth, axis=1)
    # A full-size search with the old fixed 5 px refinement window came out at
    # 0.38-0.88 px rms here; refined with a window scaled to the square, 0.08-0.19.
    assert float(np.sqrt((err ** 2).mean())) < 0.25


def test_a_hand_held_24_mp_style_frame_still_measures_a_fault(tmp_path):
    """The phone moved between the reference and the test shot, as it does."""
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    kw = dict(display_size=display.size, sensor_size=(4800, 3000), sampling_ratio=3.0)
    first = SimulatedRig(display, VirtualCamera(**kw))
    moved = SimulatedRig(display, VirtualCamera(**kw, tilt_deg=3.5, roll_deg=1.5, seed=5))
    session = CaptureSession(tmp_path, pattern_size=PATTERN, square_px=SQUARE_PX,
                             pattern_origin=PATTERN_ORIGIN, display_size=display.size,
                             values=VALUES)
    first.show("checkerboard")
    assert session.handle("calibrate", jpeg(first.read())).verdict == "OK"
    first.show("main")
    session.handle("reference", jpeg(first.read()))
    moved.show("main")
    assert session.handle("validate", jpeg(moved.read())).verdict == "PASS"

    display.offsets["TELLTALE_OIL_PRESSURE"] = (2.0, 0.0)
    report = session.handle("validate", jpeg(moved.read())).report
    hit = min(report["elements"], key=lambda e: sum(
        abs(float(v) - t) for v, t in zip(e["element_id"].split("@")[1].split(","), (216, 74))))
    assert hit["verdict"] != "PASS"
    assert 1.5 < hit["measurement"]["abs_delta"] < 2.3

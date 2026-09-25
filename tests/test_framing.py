"""What the photograph actually contains of the display.

The display's right side came back black, more on every attempt and in every
calibration mode. Two things produce that: undistortion cropping the photo to
its all-valid region -- lopsidedly when the lens solve's centre is off -- and a
test shot aimed differently from the reference. The first no longer happens;
the second is said, and what is out of frame is left out rather than failed.
"""

from __future__ import annotations

import cv2
import numpy as np

from layoutval.calibration import (
    Calibration,
    DisplayGeometry,
    DriftTracker,
    Intrinsics,
    UNSOLVED,
    feature_homography,
)
from layoutval.server import KEEP_WHOLE_PHOTO, CaptureSession, _payload
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera
from tests.conftest import NOMINAL, PATTERN, PATTERN_ORIGIN, SQUARE_PX, VALUES


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


def test_undistortion_keeps_the_whole_photo():
    """A loose phone solve, centre 6% off: cropping lost up to a quarter of one side."""
    w, h = 4032, 3024
    K = np.array([[3000.0, 0, w / 2 * 1.06], [0, 3000.0, h / 2], [0, 0, 1]])
    intr = Intrinsics(K=K, dist=np.array([0.12, -0.2, 0, 0, 0]), image_size=(w, h),
                      alpha=KEEP_WHOLE_PHOTO)
    _, _, new_K = intr.undistort_maps()
    edge = np.array([[[x, y]] for x in np.linspace(0, w - 1, 60) for y in (0.0, h - 1.0)]
                    + [[[x, y]] for y in np.linspace(0, h - 1, 60) for x in (0.0, w - 1.0)])
    moved = cv2.undistortPoints(edge, intr.K, intr.dist, P=new_K).reshape(-1, 2)
    assert (moved[:, 0] >= -1).all() and (moved[:, 0] <= w).all()
    assert (moved[:, 1] >= -1).all() and (moved[:, 1] <= h).all()
    # And the setting travels with the solve, so a saved calibration stays true.
    assert Intrinsics.from_dict(intr.to_dict()).alpha == KEEP_WHOLE_PHOTO
    assert Intrinsics.from_dict({k: v for k, v in intr.to_dict().items() if k != "alpha"}).alpha == 0.0


def test_the_capture_session_keeps_the_whole_photo(tmp_path):
    intr = Intrinsics(K=np.eye(3) * [1000, 1000, 1] + [[0, 0, 800], [0, 0, 450], [0, 0, 0]],
                      dist=np.zeros(5), image_size=(1600, 900))
    CaptureSession(tmp_path, display_size=(960, 360),
                   calibration=Calibration(intrinsics=intr, geometry=DisplayGeometry(
                       H=np.eye(3), method=UNSOLVED, display_size=(960, 360))))
    assert intr.alpha == KEEP_WHOLE_PHOTO


def test_features_find_the_pose_however_far_the_phone_moved():
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur(rng.integers(0, 255, (1100, 1960), dtype=np.uint8), (0, 0), 2)
    ref = big[100:1000, 100:1600]
    live = big[100:1000, 360:1860]                  # aimed 260 px to the right
    H = feature_homography(ref, live)
    assert H is not None
    assert abs(H[0, 2] + 260) < 1.0 and abs(H[1, 2]) < 1.0
    tracker = DriftTracker(ref, (0, 0, ref.shape[1], ref.shape[0]), motion=cv2.MOTION_HOMOGRAPHY)
    est = tracker.measure(live, init=H)
    assert est.converged
    assert abs(est.warp_camera[0, 2] + 260) < 0.3


def test_what_the_test_shot_leaves_out_is_named_not_failed(tmp_path):
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    rig = SimulatedRig(display, VirtualCamera(display_size=display.size))
    session = CaptureSession(tmp_path, pattern_size=PATTERN, square_px=SQUARE_PX,
                             pattern_origin=PATTERN_ORIGIN, display_size=display.size,
                             values=VALUES)
    rig.show("checkerboard")
    session.handle("calibrate", jpeg(rig.read()))
    rig.show("main")
    session.handle("reference", jpeg(rig.read()))
    full = rig.read()
    w = full.shape[1]
    aimed = np.zeros_like(full)
    aimed[:, :w - 260] = full[:, 260:]                # the display's left side out of frame
    rec = session.handle("validate", jpeg(aimed))
    report = rec.report
    assert not any(e["verdict"] == "FAIL" for e in report["elements"])
    flag = next(f for f in report["flags"] if f["flag"] == "outside_photo")
    assert flag["elements"] and "left" in flag["detail"]
    assert any("outside this photo" in f for f in _payload(rec)["flags"])

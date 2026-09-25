"""Camera shake: a photograph smeared by the hand moving during the exposure.

Moving the phone *between* shots is handled by re-solving the pose. This is
blur *within* one shot. Before it was handled, a 6 px streak put up to 19
elements of a good screen at FAIL, "wrong content", and a 12 px streak most of
them: correlation reads a sharp telltale against a smeared one as different.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval.blur import _spread, match_shake
from layoutval.server import CaptureSession
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera
from tests.conftest import NOMINAL, PATTERN, PATTERN_ORIGIN, SQUARE_PX, VALUES


def streak(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length + 3, length + 3), np.float32)
    c = (length + 2) / 2.0
    dx = np.cos(np.radians(angle_deg)) * length / 2.0
    dy = np.sin(np.radians(angle_deg)) * length / 2.0
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return k / k.sum()


def scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((600, 900, 3), 14, np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(20, 820)), int(rng.integers(40, 560))
        cv2.putText(img, str(int(rng.integers(10, 99))), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (230, 230, 230), 2)
    return img


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("length,angle", [(6, 20), (12, 70), (16, 135)])
def test_the_smear_is_worked_out_from_the_two_photos(length, angle):
    sharp = scene()
    k = streak(length, angle)
    smeared = cv2.filter2D(sharp, -1, k)
    forward = match_shake(sharp, smeared)
    assert forward.applied_to == "reference"
    assert forward.spread_px == pytest.approx(_spread(k.astype(np.float64)), rel=0.25)
    # Either photo may be the shaken one.
    assert match_shake(smeared, sharp).applied_to == "live"


def test_two_sharp_photos_are_left_alone():
    img = scene()
    noisy = np.clip(img + np.random.default_rng(1).normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    m = match_shake(img, noisy)
    assert m.kernel is None and m.reference is img and m.live is noisy


def test_matching_the_smear_never_moves_anything():
    """Re-centred on its own centroid, the kernel can blur an element, not shift it."""
    sharp = np.zeros((400, 400, 3), np.uint8)
    cv2.circle(sharp, (200, 200), 6, (255, 255, 255), -1)
    for x, y in ((60, 80), (330, 90), (80, 320), (310, 300)):
        cv2.rectangle(sharp, (x, y), (x + 14, y + 9), (255, 255, 255), -1)
    k = np.zeros((21, 21), np.float32)
    k[10, 4:17] = 1.0
    k[10, 16] = 4.0         # lingers at one end: the smear's centroid is off centre
    smeared = cv2.filter2D(sharp, -1, k / k.sum())
    m = match_shake(sharp, smeared)
    assert m.kernel is not None
    ys, xs = np.mgrid[0:m.kernel.shape[0], 0:m.kernel.shape[1]]
    assert float((m.kernel * xs).sum()) == pytest.approx((m.kernel.shape[1] - 1) / 2, abs=0.05)
    assert float((m.kernel * ys).sum()) == pytest.approx((m.kernel.shape[0] - 1) / 2, abs=0.05)


@pytest.mark.parametrize("length,angle", [(8, 30), (12, 110)])
def test_a_shaken_photo_of_a_good_screen_passes_and_a_fault_still_shows(tmp_path, length, angle):
    def run(sub, fault):
        display = ClusterDisplay()
        display.state.update(NOMINAL)
        rig = SimulatedRig(display, VirtualCamera(display_size=display.size))
        session = CaptureSession(tmp_path / sub, pattern_size=PATTERN, square_px=SQUARE_PX,
                                 pattern_origin=PATTERN_ORIGIN, display_size=display.size,
                                 values=VALUES)
        rig.show("checkerboard")
        session.handle("calibrate", jpeg(rig.read()))
        rig.show("main")
        session.handle("reference", jpeg(rig.read()))
        if fault:
            display.offsets["TELLTALE_OIL_PRESSURE"] = (2.0, 0.0)
        shaken = cv2.filter2D(rig.read(), -1, streak(length, angle))
        return session.handle("validate", jpeg(shaken)).report

    assert run("good", False)["verdict"] == "PASS"
    report = run("fault", True)
    oil = min(report["elements"], key=lambda e: sum(
        abs(float(v) - t) for v, t in zip(e["element_id"].split("@")[1].split(","), (216, 74))))
    assert oil["verdict"] != "PASS"
    assert 1.4 < oil["measurement"]["abs_delta"] < 2.3

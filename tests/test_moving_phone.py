"""A phone moved between shots, with haze on the glass and a room behind the screen.

What the bench showed: the same screen, the phone moved, and FAIL -- 54 of 109
elements, up to 11 px -- under a haze of room light across the glass. And a
REVIEW with all 109 passing, from display corners that took in part of the
laptop's lid and the wall behind it, which slide against the screen when the
phone moves.

The cluster here is drawn for the test -- two dials, their ticks and numbers,
a column of text -- in a window on a laptop screen, in front of a cluttered
wall at 1.5 times the distance.
"""

from __future__ import annotations

import copy
import pathlib
import tempfile

import cv2
import numpy as np
import pytest

from layoutval import anchor, residual
from layoutval.calibration import UNSOLVED, Calibration, DisplayGeometry
from layoutval.server import CaptureSession
from layoutval.simulator import Reflection, VirtualCamera
from layoutval.types import (
    ElementKind,
    ElementResult,
    ElementSpec,
    Measurement,
    PositionModel,
    ResidualFinding,
    RunReport,
    Tolerance,
    Verdict,
)

SCREEN = (1280, 800)
BEZEL = 40
WINDOW = (40, 110)                     # where the cluster's window sits on the screen
TEXT_BLOCK = (560, 180, 120, 40)       # "550 km", in the cluster's own px


def cluster(*, moved_block=(0.0, 0.0), extra=False):
    """A generic cluster: dials, ticks, numbers, a column of text."""
    w, h = 1200, 560
    img = np.full((h, w, 3), (12, 9, 7), np.uint8)
    white, blue = (235, 235, 235), (230, 170, 90)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for cx in (260, 940):
        # The dial's face: a wide soft glow, as clusters draw them.
        r = np.hypot(xx - cx, yy - 290)
        glow = np.clip(1.0 - r / 230.0, 0, 1) ** 1.5
        img = np.maximum(img, (glow[..., None] * np.array([120, 70, 30])).astype(np.uint8))
    for cx, top in ((260, 220), (940, 8)):
        cy = 290
        cv2.circle(img, (cx, cy), 205, (90, 70, 50), 2, cv2.LINE_AA)
        for k in range(28):
            a = np.radians(215 - k * 250 / 27)
            r0, r1 = (175, 202) if k % 3 == 0 else (188, 202)
            p0 = (int(cx + r0 * np.cos(a)), int(cy - r0 * np.sin(a)))
            p1 = (int(cx + r1 * np.cos(a)), int(cy - r1 * np.sin(a)))
            cv2.line(img, p0, p1, white, 5 if k % 3 == 0 else 3, cv2.LINE_AA)
        for k, value in enumerate(np.linspace(0, top, 10)):
            a = np.radians(215 - k * 250 / 9)
            p = (int(cx + 145 * np.cos(a)) - 18, int(cy - 145 * np.sin(a)) + 8)
            cv2.putText(img, f"{value:.0f}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2, cv2.LINE_AA)
        cv2.putText(img, "118" if cx < 600 else "3.5", (cx - 60, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, white, 4, cv2.LINE_AA)
        cv2.putText(img, "km/h" if cx < 600 else "x1000", (cx - 35, cy + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2, cv2.LINE_AA)
    for i, line in enumerate(("Trip   0.0 km", "Timer  0:00", "Avg.   --.- ")):
        cv2.putText(img, line, (500, 290 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
    x, y, bw, bh = TEXT_BLOCK
    block = np.zeros((bh, bw, 3), np.uint8)
    cv2.putText(block, "550 km", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, white, 2, cv2.LINE_AA)
    M = np.float32([[1, 0, moved_block[0]], [0, 1, moved_block[1]]])
    block = cv2.warpAffine(block, M, (bw, bh), flags=cv2.INTER_CUBIC)
    img[y:y + bh, x:x + bw] = np.maximum(img[y:y + bh, x:x + bw], block)
    if extra:                          # something drawn where nothing was
        cv2.circle(img, (600, 470), 16, (40, 40, 230), -1, cv2.LINE_AA)
    return img


def panel(**kw):
    w, h = SCREEN
    out = np.full((h + 2 * BEZEL, w + 2 * BEZEL, 3), 12, np.uint8)
    screen = np.full((h, w, 3), (38, 36, 40), np.uint8)
    c = cluster(**kw)
    x, y = WINDOW
    screen[y:y + c.shape[0], x:x + c.shape[1]] = c
    out[BEZEL:BEZEL + h, BEZEL:BEZEL + w] = screen
    return out


def _wall():
    rng = np.random.default_rng(3)
    wall = cv2.GaussianBlur(rng.integers(0, 255, (2600, 3600), dtype=np.uint8), (0, 0), 9)
    wall = cv2.normalize(wall, None, 110, 200, cv2.NORM_MINMAX)
    # In focus, as a room is at arm's length: shelves, boxes, frames.
    for _ in range(120):
        x, y = int(rng.integers(0, 3600)), int(rng.integers(0, 2600))
        cv2.rectangle(wall, (x, y), (x + int(rng.integers(30, 250)), y + int(rng.integers(30, 250))),
                      int(rng.integers(20, 240)), -1)
    # And right beside the laptop, where corners set a little wide reach: a
    # cable down the wall and the edge of a shelf.
    cv2.line(wall, (2525, 0), (2525, 2600), 25, 12)
    cv2.line(wall, (2560, 0), (2560, 2600), 230, 8)
    cv2.line(wall, (0, 1770), (3600, 1770), 25, 12)
    return cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)


WALL = _wall()
HAZE_REF = [Reflection((0.45, 0.40), (1.3, 1.1), 55), Reflection((0.55, 0.25), (0.25, 0.35), 70, softness=8)]
HAZE_LIVE = [Reflection((0.60, 0.45), (1.3, 1.1), 65), Reflection((0.35, 0.30), (0.25, 0.35), 80, softness=8)]


class Bench:
    def __init__(self):
        pw, ph = SCREEN[0] + 2 * BEZEL, SCREEN[1] + 2 * BEZEL
        self.cam = VirtualCamera(display_size=(pw, ph), sensor_size=(2400, 1800), sampling_ratio=1.3,
                                 tilt_deg=3.0, roll_deg=-1.0, k1=0.0, k2=0.0, seed=1,
                                 pwm_amplitude=0.0)
        bh, bw = WALL.shape[:2]
        centre = np.array([[1, 0, -(bw - pw) / 2], [0, 1, -(bh - ph) / 2], [0, 0, 1.0]])
        self.H_wall = self.cam.H_true @ centre
        self.drawing = {}

    def shoot(self, glare=()):
        p = panel(**self.drawing)
        cam = copy.copy(self.cam)
        cam.glare = list(glare)
        front = cam.shoot(p).astype(np.float32)
        back_cam = copy.copy(self.cam)
        back_cam.H_true, back_cam.glare = self.H_wall, []
        back = back_cam.shoot(WALL).astype(np.float32)
        cut = copy.copy(self.cam)
        cut.noise_sigma, cut.glare = 0.0, []
        alpha = cut.shoot(np.full(p.shape, 255, np.uint8)).astype(np.float32) / 255
        self.cam._frame += 1
        return np.clip(front * alpha + back * (1 - alpha), 0, 255).astype(np.uint8)

    def move(self, turn=(0.0, 0.0, 0.0), t=(0.0, 0.0, 0.0), depth=1.5):
        K = self.cam.K
        rx, ry, rz = np.radians(turn)
        R = (np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
             @ np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
             @ np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]))
        n, tt = np.array([[0, 0, 1.0]]), np.array(t, float).reshape(3, 1)

        def plane(d):
            return K @ (R + tt @ n / d) @ np.linalg.inv(K)

        self.cam = copy.copy(self.cam)
        self.cam._rng = np.random.default_rng(int(self.cam._frame) + 7)
        self.cam.H_true = plane(1.0) @ self.cam.H_true
        self.H_wall = plane(depth) @ self.H_wall


def jpeg(img):
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tobytes()


def start(bench, glare=(), *, widen=False):
    session = CaptureSession(
        pathlib.Path(tempfile.mkdtemp()), display_size=SCREEN, mark_corners=True, calib_mode="auto",
        calibration=Calibration(intrinsics=None, geometry=DisplayGeometry(
            H=np.eye(3), method=UNSOLVED, display_size=SCREEN)))
    img = bench.shoot(glare)
    data = jpeg(img)
    h, w = img.shape[:2]
    corners = [[x * w, y * h] for x, y in session.handle("propose", data).proposal["corners"]]
    if widen:
        # Past the display on two sides, over the lid and the wall behind it.
        corners[1][0] += 0.07 * w
        corners[2][0] += 0.07 * w
        corners[2][1] += 0.05 * h
        corners[3][1] += 0.05 * h
    session.handle("corners", data, corners=corners)
    assert session.handle("reference", data).verdict == "OK"
    return session


def _moved(report):
    """The elements measured as having moved: the fault, and nothing else."""
    return [e for e in report["elements"] if e["measurement"]["abs_delta"] > 1.0]


@pytest.mark.parametrize("move", [dict(turn=(0, 4, 0)), dict(t=(0.08, 0, 0))],
                         ids=["turned 4 deg", "4 cm sideways"])
def test_a_moved_phone_under_haze_passes_a_good_screen_and_finds_a_fault(move):
    bench = Bench()
    session = start(bench, HAZE_REF)
    bench.move(**move)
    report = session.handle("validate", jpeg(bench.shoot(HAZE_LIVE))).report
    assert report["verdict"] == "PASS", [
        (e["element_id"], e["reason"], e["measurement"]["abs_delta"])
        for e in report["elements"] if e["verdict"] != "PASS"]

    bench.drawing = dict(moved_block=(2.0, 0.0))
    report = session.handle("validate", jpeg(bench.shoot(HAZE_LIVE))).report
    moved = _moved(report)
    assert moved and all(e["verdict"] != "PASS" for e in moved)
    assert all(1.5 < e["measurement"]["abs_delta"] < 2.5 for e in moved)
    assert len(moved) <= 3                          # the block, and only the block


def test_corners_over_the_lid_and_wall_do_not_make_a_good_screen_review():
    bench = Bench()
    session = start(bench, widen=True)
    bench.move(t=(0.08, 0, 0))
    report = session.handle("validate", jpeg(bench.shoot())).report
    assert report["verdict"] == "PASS"
    assert not report["residual"]["findings"]


def test_something_drawn_where_nothing_was_is_still_review():
    bench = Bench()
    session = start(bench)
    bench.move(turn=(0, 3, 0))
    bench.drawing = dict(extra=True)
    report = session.handle("validate", jpeg(bench.shoot())).report
    assert all(e["verdict"] == "PASS" for e in report["elements"])
    assert report["residual"]["findings"]
    assert report["verdict"] == "REVIEW"


# -- the pieces -----------------------------------------------------------------


def _report_with(points, shifts, kind=ElementKind.REGION):
    specs, report = [], RunReport(screen="t")
    for i, ((x, y), (dx, dy)) in enumerate(zip(points, shifts)):
        spec = ElementSpec(id=f"e{i}", kind=kind, bbox=(x - 5, y - 5, 10, 10),
                           position=PositionModel(origin=(x - 5, y - 5)), tolerance=Tolerance())
        specs.append(spec)
        report.results.append(ElementResult(
            element_id=spec.id, verdict=Verdict.PASS, reason=None,
            measurement=Measurement(element_id=spec.id, dx=dx, dy=dy), tolerance=Tolerance()))
    return specs, report


def test_the_elements_give_the_pose_error_and_leave_a_real_fault_out_of_it():
    rng = np.random.default_rng(0)
    points = rng.uniform([20, 20], [1260, 780], (60, 2))
    error = np.array([[1.002, 0.003, 1.5], [-0.002, 0.998, -0.8], [2e-6, -1e-6, 1.0]])
    found = cv2.perspectiveTransform(points.reshape(-1, 1, 2), error).reshape(-1, 2)
    shifts = found - points + rng.normal(0, 0.05, points.shape)
    shifts[7] += (2.0, 0.0)                         # one element really moved
    specs, report = _report_with(points, shifts)
    fix = anchor.element_correction(report, specs)
    assert fix is not None and fix.inliers == 59
    # Where each element was found, taken back through the correction: the
    # pose error gone, the real move kept.
    seen = points + shifts
    left = cv2.perspectiveTransform(seen.reshape(-1, 1, 2), np.linalg.inv(fix.G)).reshape(-1, 2)
    assert np.linalg.norm(left[7] - points[7] - [2.0, 0.0]) < 0.2
    assert np.linalg.norm(np.delete(left - points, 7, axis=0), axis=1).max() < 0.2


def _two_dials(width=1920, height=1200):
    """Element centres laid out like a cluster: a dial each side, a column of
    text between them, a few status items along the top. True for the left dial."""
    points, left = [], []
    for cx, is_left in ((0.2 * width, True), (0.8 * width, False)):
        for a in np.linspace(0.75 * np.pi, 2.25 * np.pi, 26):     # ticks and numbers
            for r in (0.13 * width, 0.105 * width):
                points.append((cx + r * np.cos(a), 0.55 * height - r * np.sin(a)))
                left.append(is_left)
        for dx, dy in ((0, 0), (0, 40), (-30, 90), (30, 90)):
            points.append((cx + dx, 0.55 * height + dy))
            left.append(is_left)
    for y in np.linspace(0.35, 0.8, 9):
        for x in (0.42, 0.55):
            points.append((x * width, y * height))
            left.append(False)
    for x in (0.47, 0.5, 0.53, 0.97, 0.985):
        points.append((x * width, 0.03 * height))
        left.append(False)
    return np.array(points), np.array(left)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("move", [(4.0, 0.0), (0.0, 4.0), (-3.0, 3.0)])
def test_a_dial_that_moved_as_a_whole_is_left_out_whole(move, seed):
    """The whole left dial drawn a few pixels off, nothing else.

    Nothing but the dial pins that side of the screen, so a homography can bend
    to follow it; counting inliers preferred that, because it explains more
    elements. Then half the dial or none of it failed, and elements elsewhere
    failed instead -- on this layout 0-21 of the dial's 56 elements, and 2-33 of
    the 79 that had not moved.
    """
    rng = np.random.default_rng(seed)
    points, dial = _two_dials()
    error = np.array([[1.002, 0.003, 1.5], [-0.002, 0.998, -0.8], [2e-6, -1e-6, 1.0]])
    found = cv2.perspectiveTransform(points.reshape(-1, 1, 2), error).reshape(-1, 2)
    noise = rng.normal(0, 0.25, points.shape)
    shifts = found - points + noise
    shifts[dial] += move
    specs, report = _report_with(points, shifts)
    fix = anchor.element_correction(report, specs)
    assert fix is not None and fix.grouped == dial.sum()
    seen = points + shifts
    left = cv2.perspectiveTransform(seen.reshape(-1, 1, 2), np.linalg.inv(fix.G)).reshape(-1, 2)
    # Every element of the dial shows the move and every other none, give or
    # take its own measurement noise and what that noise leaves in a pose fitted
    # to 135 elements (0.13-0.40 px here).
    error = left - points - noise
    assert np.linalg.norm(error[dial] - move, axis=1).max() < 0.5
    assert np.linalg.norm(error[~dial], axis=1).max() < 0.5


def _labels(offsets):
    """A dark screen of distinct labels, each drawn at its place plus its offset."""
    img = np.full((SCREEN[1], SCREEN[0], 3), 25, np.uint8)
    rng = np.random.default_rng(5)
    specs = []
    for i in range(60):
        col, row = i % 10, i // 10
        text = "".join(rng.choice(list("ABDEFHKMNPRSTUVWXYZ0123456789"), 3))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x, y = 20 + col * 125, 60 + row * 125
        dx, dy = offsets(x)
        cv2.putText(img, text, (int(x + dx), int(y + th + dy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (220, 220, 220), 2, cv2.LINE_AA)
        box = (x - 4.0, y - 4.0, tw + 8.0, th + 14.0)
        specs.append(ElementSpec(id=f"l{i}", kind=ElementKind.REGION, bbox=box,
                                 position=PositionModel(origin=box[:2]), tolerance=Tolerance()))
    return img, specs


def test_every_element_votes_even_when_the_first_pose_followed_the_moved_group():
    """The first pose locked onto a group that moved -- the sharpest thing on a
    hazy screen -- so the rest sat 12 px off, past the usual 8 px search, came
    back missing and had no say; the moved group passed and the rest failed.
    Looked for far enough, the rest outvote it."""
    group = lambda x: x < 0.35 * SCREEN[0]
    reference, specs = _labels(lambda x: (0.0, 0.0))
    live, _ = _labels(lambda x: (0.0, 0.0) if group(x) else (-12.0, 0.0))
    fix = anchor.far_correction(reference, live, specs)
    moved = np.array([group(s.bbox[0]) for s in specs])
    assert fix is not None
    assert fix.grouped == moved.sum() and fix.inliers == (~moved).sum()
    # the mapping is the rest's 12 px, so the group is what shows a move
    centre = cv2.perspectiveTransform(np.float64([[[640.0, 400.0]]]), fix.G).ravel()
    assert np.allclose(centre, [628.0, 400.0], atol=0.5)


def _small_cluster():
    """32 element centres on a 1920 x 1200 screen, laid out like a sparse
    cluster: a speed band of 11 down the left with the speed digits beside it,
    a band of 8 down the right with the gear letter beside it, a media tile in
    the middle, a row along the bottom. Returns the centres and what moves."""
    pts, what = [], []
    for y in np.linspace(300, 800, 11):
        pts.append((290 + 20 * ((y // 100) % 2), y)); what.append("band")
    pts += [(230, 560), (240, 620)]; what += ["digits", "unit"]
    for y in np.linspace(320, 780, 8):
        pts.append((1660, y)); what.append("right")
    pts.append((1750, 560)); what.append("gear")
    pts += [(960, 500), (960, 600), (940, 640), (1000, 700)]; what += ["media"] * 4
    pts += [(150, 870), (300, 880), (960, 860), (1620, 880), (1700, 880), (1760, 890)]
    what += ["bottom"] * 6
    return np.array(pts, float), np.array(what)


def test_a_small_cluster_with_three_things_moved_still_anchors_on_the_rest():
    """Speed band 4 px, speed digits and gear letter 3 px: 13 of 32 moved.

    That left 19 to fit, one short of the 20 a homography was allowed from,
    so no correction was made and the pose stayed where ECC put it -- pulled
    1-2 px towards what moved, so the band came out half green, half red. It
    worked with the gear letter left alone, because then 20 were left.
    """
    rng = np.random.default_rng(3)
    points, what = _small_cluster()
    error = np.array([[1.001, 0.002, 1.2], [-0.001, 0.999, -0.6], [1e-6, -5e-7, 1.0]])
    found = cv2.perspectiveTransform(points.reshape(-1, 1, 2), error).reshape(-1, 2)
    noise = rng.normal(0, 0.15, points.shape)
    shifts = found - points + noise
    move = {"band": (4.0, 0.0), "digits": (3.0, 0.0), "gear": (3.0, 0.0)}
    for name, d in move.items():
        shifts[what == name] += d
    specs, report = _report_with(points, shifts)
    fix = anchor.element_correction(report, specs)
    assert fix is not None
    left = cv2.perspectiveTransform((points + shifts).reshape(-1, 1, 2),
                                    np.linalg.inv(fix.G)).reshape(-1, 2) - points - noise
    for name in np.unique(what):
        expected = np.array(move.get(name, (0.0, 0.0)))
        assert np.linalg.norm(left[what == name] - expected, axis=1).max() < 0.5, name


def test_too_few_elements_are_not_anchored_on():
    specs, report = _report_with([(100, 100), (500, 120), (300, 400)], [(1, 0)] * 3)
    assert anchor.element_correction(report, specs) is None
    assert not anchor.can_anchor(specs)


def test_a_needle_is_not_anchored_on():
    points = [(100 + 100 * i, 100 + 60 * (i % 3)) for i in range(12)]
    specs, report = _report_with(points, [(1.0, 0.0)] * 12, kind=ElementKind.NEEDLE)
    assert anchor.element_correction(report, specs) is None


def test_shifted_content_is_set_aside_and_drawn_content_is_not():
    rng = np.random.default_rng(1)
    ref = cv2.GaussianBlur(rng.integers(0, 255, (300, 400), dtype=np.uint8), (0, 0), 2)
    shifted = np.roll(ref, (0, 9), axis=(0, 1))
    assert residual.displaced(ref, shifted, (100, 100, 60, 40), reach=16)
    drawn = ref.copy()
    cv2.circle(drawn, (130, 120), 18, 255, -1)
    flat = np.full_like(ref, 20)
    assert not residual.displaced(flat, drawn, (110, 100, 40, 40), reach=16)
    assert not residual.displaced(ref, drawn, (110, 100, 40, 40), reach=16)

    report = RunReport(screen="t")
    report.residual_findings = [ResidualFinding(bbox=(100, 100, 60, 40), mean_dissimilarity=0.8,
                                                area_px=2400, overlaps=[])]
    assert residual.set_aside_displaced(report, ref, shifted, reach=16) == 1
    assert not report.residual_findings
    assert any(f["flag"] == "residual_displaced" for f in report.flags)


def test_off_the_drawing_only_detail_the_reference_had_is_set_aside():
    ref = np.full((400, 600), 12, np.uint8)
    cv2.line(ref, (560, 0), (560, 400), 200, 6)            # a cable beside the screen
    drawn = np.zeros((400, 600), bool)
    drawn[50:350, 50:450] = True                            # where the cluster draws
    report = RunReport(screen="t")
    report.residual_findings = [
        ResidualFinding(bbox=(540, 0, 40, 400), mean_dissimilarity=0.9, area_px=16000),
        ResidualFinding(bbox=(480, 360, 30, 30), mean_dissimilarity=0.9, area_px=900),
        ResidualFinding(bbox=(200, 200, 30, 30), mean_dissimilarity=0.9, area_px=900),
    ]
    assert residual.set_aside_beyond(report, ref, drawn) == 1
    # Kept: something on plain background past the drawing, and inside it.
    assert [f.bbox for f in report.residual_findings] == [(480, 360, 30, 30), (200, 200, 30, 30)]

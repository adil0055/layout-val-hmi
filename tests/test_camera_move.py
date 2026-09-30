"""The phone moved between the reference and the test shot, in a real scene.

A real photograph is not only the screen. There is a desk, a keyboard, a
dashboard round the cluster -- at other depths, so when the camera moves they
shift against the screen (parallax). The pose used to be fitted to the whole
photograph: moved 3 cm sideways at 50 cm with a backdrop 40% further away, a
screen with nothing wrong came back FAIL on every element, 6-11 px out. It is
now fitted to the screen alone.

And the one fault a pose re-solved from the screen cannot see -- the whole
layout moved -- is checked against the display's own edges with ``--edges``.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from layoutval import edgecheck
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera
from tests.test_modes import NOMINAL, jpeg, make_session

OIL = (216, 74)


def _clutter(size=(4400, 3200), seed=3):
    rng = np.random.default_rng(seed)
    w, h = size
    bg = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 6)
    bg = cv2.normalize(bg, None, 20, 200, cv2.NORM_MINMAX)
    for _ in range(60):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(bg, (x, y), (x + int(rng.integers(40, 300)), y + int(rng.integers(40, 300))),
                      int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


BACKDROP = _clutter()


class DepthRig(BezelRig):
    """The bezel rig in front of a cluttered backdrop on a plane of its own."""

    def __init__(self, display, camera):
        super().__init__(display, BezelPanel(), camera)
        pw, ph = self.panel.panel_size
        bh, bw = BACKDROP.shape[:2]
        centre = np.array([[1, 0, -(bw - pw) / 2], [0, 1, -(bh - ph) / 2], [0, 0, 1]], float)
        self.H_backdrop = camera.H_true @ centre

    def read(self):
        panel = self.panel.render(self.display.render())
        front = self.camera.shoot(panel).astype(np.float32)
        back_cam = copy.copy(self.camera)
        back_cam.H_true = self.H_backdrop
        back = back_cam.shoot(BACKDROP).astype(np.float32)
        cut = copy.copy(self.camera)
        cut.noise_sigma, cut.pwm_amplitude, cut.glare = 0.0, 0.0, []
        alpha = cut.shoot(np.full(panel.shape, 255, np.uint8)).astype(np.float32) / 255.0
        return np.clip(front * alpha + back * (1 - alpha), 0, 255).astype(np.uint8)

    def move(self, *, turn_deg=(0.0, 0.0, 0.0), shift=(0.0, 0.0, 0.0), depth=1.4):
        """Rotate and translate the camera; ``shift`` in units of the screen's distance."""
        K = self.camera.K
        rx, ry, rz = np.radians(turn_deg)
        R = (np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
             @ np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
             @ np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]))
        t = np.array(shift, float).reshape(3, 1)
        n = np.array([[0.0, 0.0, 1.0]])

        def plane(d):
            return K @ (R + t @ n / d) @ np.linalg.inv(K)

        moved = copy.copy(self.camera)
        moved._rng = np.random.default_rng(9)
        moved.H_true = plane(1.0) @ self.camera.H_true
        self.H_backdrop = plane(depth) @ self.H_backdrop
        self.camera = moved


def tilt_screen(rig, up=0.25, out=0.10):
    """The screen tilts on a hinge along its bottom edge -- a laptop's lid -- and
    nothing else moves: the top edge comes up and widens, the backdrop stays."""
    pw, ph = rig.panel.panel_size
    cam = rig.camera
    q = cv2.perspectiveTransform(
        np.float64([[[0, 0]], [[pw, 0]], [[pw, ph]], [[0, ph]]]), cam.H_true).reshape(4, 2)
    w, h = q[1, 0] - q[0, 0], q[3, 1] - q[0, 1]
    n = q.copy()
    n[0] += (-out * w, -up * h)
    n[1] += (out * w, -up * h)
    moved = copy.copy(cam)
    moved._rng = np.random.default_rng(9)
    moved.H_true = cv2.getPerspectiveTransform(q.astype(np.float32), n.astype(np.float32)) @ cam.H_true
    rig.camera = moved


@pytest.fixture()
def rig():
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    return DepthRig(display, VirtualCamera(
        display_size=BezelPanel().panel_size, sensor_size=(2400, 1500),
        sampling_ratio=0.62, tilt_deg=4.0, roll_deg=-2.0, seed=1))


def _start(tmp_path, rig, **kw):
    session = make_session(tmp_path, rig)
    for key, value in kw.items():
        setattr(session, key, value)
    frame = rig.read()
    data = jpeg(frame)
    h, w = frame.shape[:2]
    proposal = session.handle("propose", data).proposal
    session.handle("corners", data, corners=[[x * w, y * h] for x, y in proposal["corners"]])
    assert session.handle("reference", data).verdict == "OK"
    return session


def _nearest(report, xy):
    return min(report["elements"], key=lambda e: sum(
        abs(float(v) - t) for v, t in zip(e["element_id"].split("@")[1].split(","), xy)))


def test_moving_sideways_in_front_of_a_backdrop_still_measures_the_screen(tmp_path, rig):
    session = _start(tmp_path, rig)
    rig.move(shift=(0.06, 0.0, 0.0))                  # 3 cm at 50 cm
    report = session.handle("validate", jpeg(rig.read())).report
    assert report["verdict"] == "PASS"
    assert max(e["measurement"]["abs_delta"] for e in report["elements"]) < 0.8

    rig.display.offsets["TELLTALE_OIL_PRESSURE"] = (2.0, 0.0)
    report = session.handle("validate", jpeg(rig.read())).report
    hit = _nearest(report, OIL)
    assert hit["verdict"] != "PASS"
    assert 1.5 < hit["measurement"]["abs_delta"] < 2.5


def test_a_screen_tilted_on_its_own_is_followed_not_the_room(tmp_path, rig):
    """A laptop's lid tilted between the shots, the room behind it where it was.

    Matched over the whole photograph, the backdrop's points agreed that
    nothing had moved and the pose was solved from there: the hinge's corners
    right and the top ones hundreds of pixels out, nearly every element failing.
    """
    session = _start(tmp_path, rig)
    tilt_screen(rig)
    report = session.handle("validate", jpeg(rig.read())).report
    assert report["verdict"] == "PASS", [
        (e["element_id"], e["reason"]) for e in report["elements"] if e["verdict"] != "PASS"]

    rig.display.offsets["TELLTALE_OIL_PRESSURE"] = (2.0, 0.0)
    report = session.handle("validate", jpeg(rig.read())).report
    hit = _nearest(report, OIL)
    assert hit["verdict"] != "PASS"
    assert 1.4 < hit["measurement"]["abs_delta"] < 2.6


def test_a_whole_layout_shift_is_found_against_the_edges_with_edges_on(tmp_path, rig):
    session = _start(tmp_path, rig, edge_check=True)
    rig.move(turn_deg=(2.0, -3.0, 1.5), shift=(0.05, -0.03, 0.06))
    clean = session.handle("validate", jpeg(rig.read())).report
    edge = next(f for f in clean["flags"] if f["flag"] == "layout_vs_edges")
    assert edge["checked"] and edge["severity"] == "note"
    assert clean["verdict"] == "PASS"

    for eid in rig.display.LAYOUT:
        rig.display.offsets[eid] = (2.0, 0.0)
    rig.display.needle_pivot_offset = (2.0, 0.0)
    report = session.handle("validate", jpeg(rig.read())).report
    # The pose follows the screen, so no element on its own is out ...
    assert all(e["measurement"]["abs_delta"] < 1.5 for e in report["elements"])
    # ... and the edges are what say the layout moved.
    edge = next(f for f in report["flags"] if f["flag"] == "layout_vs_edges")
    assert edge["severity"] == "review"
    assert 1.6 < edge["dx"] < 2.5 and abs(edge["dy"]) < 0.4
    assert report["verdict"] != "PASS"


def test_the_edge_check_is_off_unless_asked_for(tmp_path, rig):
    session = _start(tmp_path, rig)
    rig.move(turn_deg=(0.0, 3.0, 0.0))
    report = session.handle("validate", jpeg(rig.read())).report
    assert not [f for f in report["flags"] if f["flag"] == "layout_vs_edges"]


def _screen(shift=0.0, seed=4):
    """A dim screen in a black bezel with a glint on the glass: no clean step."""
    rng = np.random.default_rng(seed)
    img = np.full((600, 800), 22, np.float32)
    cv2.rectangle(img, (93, 73), (706, 526), 34, 1)            # glint just outside
    img[80:520, 100:700] = 48                                   # the lit area: fixed
    for _ in range(25):                                         # the drawing: moves
        x, y = rng.integers(130, 640), rng.integers(110, 470)
        w, h = rng.integers(10, 40), rng.integers(10, 40)
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        blob = np.zeros_like(img)
        cv2.rectangle(blob, (int(x), int(y)), (int(x + w), int(y + h)), 180, -1)
        img = np.maximum(img, cv2.warpAffine(blob, M, img.shape[::-1], flags=cv2.INTER_LINEAR))
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("drawn_shift", [0.0, 2.0])
def test_the_edges_say_how_far_the_drawing_moved_on_a_screen_with_no_clean_step(drawn_shift):
    reference = _screen()
    camera = np.array([[1.02, 0.01, 12.0], [-0.012, 1.01, -7.0], [2e-5, -1e-5, 1.0]])
    live = cv2.warpPerspective(_screen(drawn_shift), camera, (800, 600), flags=cv2.INTER_CUBIC)
    H_ref = np.array([[1.0, 0, 100], [0, 1, 80], [0, 0, 1]])    # display -> reference photo
    # A pose solved from the drawing follows the drawing.
    pose = camera @ H_ref @ np.array([[1, 0, drawn_shift], [0, 1, 0], [0, 0, 1]]) @ np.linalg.inv(H_ref)
    # Corners a pixel or two out, as a proposal leaves them.
    corners = np.array([[98.0, 81.0], [701.0, 78.5], [699.0, 521.5], [101.5, 518.0]])
    sides = edgecheck.edge_offsets(reference, live, corners, pose)
    assert all(s is not None for s in sides)
    shift = edgecheck.layout_shift(H_ref, sides, drawn=(0.0, 0.0))
    assert shift.dx == pytest.approx(drawn_shift, abs=0.2)
    assert shift.dy == pytest.approx(0.0, abs=0.2)


def test_opposite_sides_that_disagree_are_not_a_shift():
    H_ref = np.eye(3)

    def side(offset, normal):
        return edgecheck.SideOffset(offset, 0.05, np.array(normal, float), np.array([400.0, 300.0]))

    sides = [side(0.0, (0, -1)), side(-2.0, (1, 0)), side(0.0, (0, 1)), side(+1.0, (-1, 0))]
    shift = edgecheck.layout_shift(H_ref, sides, drawn=(0.0, 0.0))
    assert shift.dx is None                 # right and left 1 px apart
    assert shift.dy == pytest.approx(0.0)

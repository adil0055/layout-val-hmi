"""Proposing the display's corners from an ordinary photograph.

The scenes here are built to contain every trap that a real bench photograph
set for the first attempts: a screen with a desktop bar and a window title bar
inside it, a bezel, a lid, a keyboard whose legends are lit marks just below
the display, and a cluttered background with straight edges of its own. Each
rule in :mod:`layoutval.displayfind` exists because one of these picked the
wrong rectangle.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval.calibration import (
    DisplayGeometry,
    Intrinsics,
    Undistorter,
    find_display_aperture,
)
from layoutval.displayfind import propose_display_corners
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera

NOMINAL = {
    "TELLTALE_BATTERY_LOW": True, "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True, "FUEL_LEVEL": 0.6, "SPEED": 120.0,
}


def bench_scene(rng: np.random.Generator, size=(2000, 1500)):
    """A laptop on a desk, running the cluster in a window. Returns (img, corners)."""
    w, h = size
    img = np.full((h, w, 3), 200, np.uint8)
    # Background clutter: panels, fittings, straight edges that are not ours.
    for _ in range(14):
        x0, y0 = rng.integers(0, w), rng.integers(0, int(h * 0.3))
        x1, y1 = x0 + rng.integers(80, 600), y0 + rng.integers(20, 200)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      tuple(int(v) for v in rng.integers(120, 250, 3)), -1)
    # The lid and its bezel. Grey levels here are measured off two bench
    # photographs, not chosen: a glossy black bezel reflects the room and sits
    # at 48-124, the photographed "black" of a lit screen at 41-126 -- the two
    # within about ten levels of each other -- and the panel's mask line runs
    # 15-25 levels darker than both. An earlier version of this scene used a
    # black screen and a near-black bezel, which is not what a camera sees, and
    # tuned the detector toward a problem nobody has.
    lid = (int(0.05 * w), int(0.08 * h), int(0.95 * w), int(0.86 * h))
    cv2.rectangle(img, lid[:2], lid[2:], (58, 58, 60), -1)
    cv2.rectangle(img, lid[:2], lid[2:], (150, 150, 150), 3)   # the lid's bright rim
    # The screen, a little inside the lid. Its background is the cluster's own:
    # on the real bench the window's content *is* the cluster, so there is no
    # inner rectangle for anything to lock onto.
    sx0, sy0 = lid[0] + 30, lid[1] + 40
    sx1, sy1 = lid[2] - 30, lid[3] - 55
    cluster = ClusterDisplay()
    cluster.state.update(NOMINAL)
    bg = tuple(int(v) for v in cluster.render()[5, 5])
    screen = np.zeros((sy1 - sy0, sx1 - sx0, 3), np.uint8)
    screen[:] = bg
    sw, sh = sx1 - sx0, sy1 - sy0
    # Desktop top bar, with a clock and icons; a window title bar under it.
    cv2.rectangle(screen, (0, 0), (sw, 26), (30, 30, 30), -1)
    # Desktop text as a camera sees it: bold enough to peak where the bench
    # photographs' text does (146-159 after the top-hat), not a hairline.
    cv2.putText(screen, "Sep 17 12:45", (sw // 2 - 60, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 2, cv2.LINE_AA)
    for k in range(3):
        cv2.circle(screen, (sw - 30 - 22 * k, 13), 5, (235, 235, 235), -1)
    cv2.rectangle(screen, (0, 26), (sw, 56), (48, 48, 48), -1)
    cv2.line(screen, (0, 56), (sw, 56), (90, 90, 90), 1)
    cv2.putText(screen, "Instrument Cluster", (sw // 2 - 80, 47), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 2, cv2.LINE_AA)
    # The cluster itself, in the window.
    body = cv2.resize(cluster.render(), (sw - 40, sh - 90))
    screen[70:70 + body.shape[0], 20:20 + body.shape[1]] = body
    # What the camera sees of a lit panel: backlight and room reflection lift
    # its black to about the bezel's level.
    screen = (screen.astype(np.float32) * 0.8 + 45).astype(np.uint8)
    img[sy0:sy1, sx0:sx1] = screen
    # The panel's black mask: the thin line round the active area that a
    # zoomed-in bench photograph shows on every side, and the feature a line
    # detector actually finds -- about twenty levels under both neighbours.
    cv2.rectangle(img, (sx0 - 2, sy0 - 2), (sx1 + 1, sy1 + 1), (30, 30, 30), 2)
    # A keyboard below, whose legends are exactly the lit marks that fooled it.
    ky0 = lid[3] + 25
    for row in range(4):
        for col in range(14):
            x = lid[0] + 20 + col * int((lid[2] - lid[0] - 40) / 14)
            y = ky0 + row * 55
            cv2.rectangle(img, (x, y), (x + 95, y + 45), (40, 40, 40), -1)
            cv2.putText(img, "QWERTYUIOPASDF"[col], (x + 38, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    corners = np.array([[sx0 - 0.5, sy0 - 0.5], [sx1 - 0.5, sy0 - 0.5],
                        [sx1 - 0.5, sy1 - 0.5], [sx0 - 0.5, sy1 - 0.5]], np.float64)
    return img, corners


def warped(img, corners, rng, max_jitter=0.05, max_angle=8.0):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-max_jitter, max_jitter, (4, 2)) * [w, h]
    a = np.radians(rng.uniform(-max_angle, max_angle))
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([w / 2, h / 2])
    dst = ((src + jit - c) @ r.T) * rng.uniform(0.8, 1.0) + c
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(img, m, (w, h), borderValue=(200, 200, 200))
    out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.2), 0, 255).astype(np.uint8)
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), m)
    return out, moved.reshape(-1, 2)


# --------------------------------------------------------------------------


def test_finds_the_screen_not_the_title_bar_lid_or_keyboard():
    """The failure it is built around, square on."""
    img, truth = bench_scene(np.random.default_rng(0))
    proposal = propose_display_corners(img)
    assert proposal is not None
    worst = float(np.linalg.norm(proposal.corners - truth, axis=1).max())
    # 12 px: the proposal anchors on the panel's mask line, a few pixels
    # outside the active area, while every wrong rectangle in this scene -- the
    # title bar, the desktop bar, the lid -- is 20 px or more away.
    assert worst < 12.0, f"worst corner {worst:.1f} px"


@pytest.mark.parametrize("seed", range(8))
def test_survives_perspective_rotation_and_exposure(seed):
    rng = np.random.default_rng(100 + seed)
    img, truth = bench_scene(rng)
    img, truth = warped(img, truth, rng)
    proposal = propose_display_corners(img)
    assert proposal is not None
    worst = float(np.linalg.norm(proposal.corners - truth, axis=1).max())
    assert worst < 0.01 * np.hypot(*img.shape[:2]), f"worst corner {worst:.1f} px"


def test_a_soft_glare_patch_does_not_move_it():
    """Glare is smooth; the lit-mark test is a top-hat, which ignores smooth."""
    rng = np.random.default_rng(3)
    img, truth = bench_scene(rng)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    blob = 140.0 * np.exp(-(((xx - 0.45 * w) / (0.12 * w)) ** 2
                            + ((yy - 0.4 * h) / (0.15 * h)) ** 2))
    glared = np.clip(img + blob[..., None], 0, 255).astype(np.uint8)
    clean = propose_display_corners(img)
    shiny = propose_display_corners(glared)
    assert clean is not None and shiny is not None
    assert float(np.linalg.norm(shiny.corners - truth, axis=1).max()) < 12.0


def test_agrees_with_the_border_fit_on_a_cluster_in_its_bezel():
    """On a real cluster the display sits in trim with nothing else around it."""
    for i, (tilt, roll) in enumerate([(2.0, 0.7), (8.0, -6.0), (-6.0, 4.0)]):
        display = ClusterDisplay()
        display.state.update(NOMINAL)
        rig = BezelRig(display, BezelPanel(), VirtualCamera(
            display_size=BezelPanel().panel_size, sensor_size=(2400, 1500),
            sampling_ratio=0.95, tilt_deg=tilt, roll_deg=roll, seed=i))
        undistort = Undistorter(Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                           image_size=rig.camera.sensor_size))
        rig.show("main")
        live = undistort(rig.read())
        border, _ = find_display_aperture(live, display_size=display.size)
        proposal = propose_display_corners(live)
        assert proposal is not None
        assert float(np.linalg.norm(proposal.corners - border, axis=1).max()) < 1.5


def test_nothing_to_find_is_none_not_a_guess():
    blank = np.full((900, 1200, 3), 128, np.uint8)
    assert propose_display_corners(blank) is None


def test_proposal_serialises_as_fractions_of_the_image():
    img, _ = bench_scene(np.random.default_rng(0))
    proposal = propose_display_corners(img)
    # Through JSON, because that is where it goes: numpy scalars do not survive it.
    import json
    d = json.loads(json.dumps(proposal.to_dict(img.shape)))
    assert len(d["corners"]) == 4
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in d["corners"])
    assert set(d["sides"]) == {"top", "bottom", "left", "right"}
    assert isinstance(d["confident"], bool)


def test_the_proposal_calibrates_through_the_marked_corners_route():
    """Proposed dots go through exactly the same path as tapped ones."""
    from layoutval.calibration import homography_from_marked_corners

    img, truth = bench_scene(np.random.default_rng(0))
    proposal = propose_display_corners(img)
    geometry = homography_from_marked_corners(img, proposal.corners,
                                              display_size=(1920, 1200))
    assert isinstance(geometry, DisplayGeometry)
    rect = geometry.rectify(img)
    assert rect.shape[:2] == (1200, 1920)

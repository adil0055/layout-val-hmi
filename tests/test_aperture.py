"""Calibration from the display's own border, with no cooperation at all.

Every other route asks the cluster for something -- a pattern drawn, a white
frame, its framebuffer, or the one binding frame the bezel markers need.  This
one asks for nothing, which is the only thing that is actually true of a
production cluster: you get a camera, and the screen shows what it shows.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval.autoprofile import profile_from_reference
from layoutval.calibration import (
    Calibration,
    DisplayGeometry,
    Intrinsics,
    Undistorter,
    find_display_aperture,
    homography_from_display_aperture,
)
from layoutval.measure import measure_translation
from layoutval.server import CaptureSession
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera

NOMINAL = {
    "TELLTALE_BATTERY_LOW": True,
    "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True,
    "FUEL_LEVEL": 0.6,
    "SPEED": 120.0,
}


def rig_for(**camera):
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    panel = BezelPanel(bezel_grey=camera.pop("bezel_grey", 38))
    opts = dict(display_size=panel.panel_size, sensor_size=(2400, 1500),
                sampling_ratio=0.95)
    opts.update(camera)
    return display, panel, BezelRig(display, panel, VirtualCamera(**opts))


def shot(rig):
    """An undistorted frame of the cluster showing only its own UI."""
    undistort = Undistorter(Intrinsics(
        K=rig.camera.K, dist=rig.camera.dist, image_size=rig.camera.sensor_size))
    rig.show("main")
    return undistort(rig.read())


def element_error(display, live, H):
    truth = display.render()
    probe = profile_from_reference(truth, screen="b", display_size=display.size)
    rect = DisplayGeometry(H=H, display_size=display.size).rectify(live)
    errors = [
        float(np.hypot(m.dx, m.dy))
        for spec in probe
        for m in [measure_translation(truth, rect, spec)]
        if m.dx is not None and m.zncc and m.zncc > 0.5
    ]
    assert errors, "nothing matched at all"
    return float(np.sqrt(np.mean(np.square(errors))))


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


# --------------------------------------------------------------------------


@pytest.mark.parametrize("tilt,roll", [(2.0, 0.7), (6.0, -3.0), (-4.0, 5.0),
                                       (11.0, 2.5), (8.0, -8.0)])
def test_measures_to_a_tenth_of_a_pixel_with_nothing_drawn(tilt, roll):
    """The headline claim, per pose, on a screen showing only the HMI."""
    display, _, rig = rig_for(tilt_deg=tilt, roll_deg=roll)
    live = shot(rig)
    geometry = homography_from_display_aperture(live, display_size=display.size)
    assert geometry.method == "display_aperture"
    assert element_error(display, live, geometry.H) < 0.3


def test_the_half_pixel_convention_is_not_off_by_half_a_pixel():
    """The aperture is the OUTER edge of pixel 0, which sits at -0.5.

    Getting this wrong is invisible -- every pose agrees with every other pose,
    and the whole thing is shifted. It was worth a measured 0.70 px, which is
    exactly half a pixel on each axis, and it took a suspiciously round number
    to notice.
    """
    display, _, rig = rig_for()
    live = shot(rig)
    good = homography_from_display_aperture(live, display_size=display.size)
    assert element_error(display, live, good.H) < 0.3

    corners, _ = find_display_aperture(live, display_size=display.size)
    w, h = display.size
    naive = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    H, _ = cv2.findHomography(naive, corners, method=0)
    # Half a pixel on each axis is 0.707, and that is what the naive convention
    # costs -- enough to swamp the measurement it is supposed to support.
    assert element_error(display, live, H) == pytest.approx(0.707, abs=0.2)


def test_refinement_failure_refuses_rather_than_answering_coarsely():
    """The one silent failure this route had, turned into a message.

    Falling back to the threshold corners looked reasonable and measured
    terribly: every frame that fell back was 1.7-3.1 px out while every refined
    one was inside 0.05 px, and nothing in the result told them apart.
    """
    display, _, rig = rig_for()
    live = shot(rig)
    corners, diagnostics = find_display_aperture(live, display_size=display.size)
    assert diagnostics["refined"] is True

    coarse, _ = find_display_aperture(live, display_size=display.size, refine=False)
    # The coarse corners exist and are usable-looking, which is the trap.
    assert coarse.shape == (4, 2)
    assert float(np.abs(coarse - corners).max()) > 0.05


def test_a_clipped_aperture_is_refused_not_guessed():
    """Part of a rectangle still fits a rectangle, confidently and wrongly."""
    display, _, rig = rig_for()
    live = shot(rig)
    # Crop into the display so its boundary runs off the frame.
    h, w = live.shape[:2]
    cropped = live[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    with pytest.raises(RuntimeError):
        homography_from_display_aperture(cropped, display_size=display.size)


def test_too_little_contrast_is_refused_not_fudged():
    """A black screen in black trim has no boundary to find, so say so."""
    display, _, rig = rig_for(bezel_grey=15)
    live = shot(rig)
    with pytest.raises(RuntimeError):
        homography_from_display_aperture(live, display_size=display.size)


def test_it_beats_the_route_that_needs_the_screen():
    """Not a claim to defend, just the thing worth knowing: it is not a fallback.

    A four-line fit over the whole display boundary averages thousands of edge
    pixels; a chessboard localises each of its corners independently. The border
    is simply more constraint.
    """
    from layoutval.calibration import (
        chessboard_display_points, homography_from_display_pattern)

    display, _, rig = rig_for()
    live = shot(rig)
    aperture = homography_from_display_aperture(live, display_size=display.size)

    undistort = Undistorter(Intrinsics(
        K=rig.camera.K, dist=rig.camera.dist, image_size=rig.camera.sensor_size))
    rig.show("checkerboard")
    board = homography_from_display_pattern(
        undistort(rig.read()), (9, 6),
        chessboard_display_points((9, 6), 40.0, (60.0, 20.0)),
        display_size=display.size)
    rig.show("main")

    assert element_error(display, live, aperture.H) <= \
        element_error(display, live, board.H) + 0.05


def test_through_the_capture_server_end_to_end(tmp_path):
    """Calibrate, reference and validate, with the screen never touched."""
    display, _, rig = rig_for()
    session = CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                  image_size=rig.camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), display_size=display.size)),
        display_size=display.size,
        aperture=True,
    )
    assert session.status()["calibrates_from"] == "the display's own border"

    rig.show("main")
    raw = rig.read()
    rec = session.handle("calibrate", jpeg(raw))
    assert rec.verdict == "OK", rec.detail
    assert "nothing was asked of the cluster" in rec.detail
    assert "constant offset" in rec.detail

    assert session.handle("reference", jpeg(raw)).verdict == "OK"
    assert session.handle("validate", jpeg(raw)).verdict in ("PASS", "REVIEW")


def test_border_route_refuses_without_intrinsics(tmp_path):
    """It fits straight lines to edges that lens distortion bows.

    Measured: 0.04-0.06 px undistorted on any lens, against 1.3 px on a mild
    one and 4-8 px on a normal phone lens. This is the route that needs
    undistortion most, which is the opposite of what it looks like.
    """
    display, _, rig = rig_for()
    session = CaptureSession(tmp_path, display_size=display.size, aperture=True)
    assert session.status()["needs_intrinsics"] is True

    rig.show("main")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "FAILED"
    assert "intrinsics" in rec.detail
    assert "Intrinsics" in rec.detail  # names the step that fixes it


def test_lens_board_is_independent_of_how_geometry_is_solved(tmp_path):
    """Intrinsics belong to the camera, not to the rig's calibration route.

    Keeping them separate is what lets the border route -- which asks the
    cluster for nothing -- still get undistortion, from a printed board held in
    your hand that never goes near the screen.
    """
    from layoutval.calibration import CharucoSpec

    display, panel, rig = rig_for()
    spec = CharucoSpec(panel.squares[0], panel.squares[1], panel.square_length,
                       panel.square_length * panel.marker_ratio)
    session = CaptureSession(tmp_path, display_size=display.size,
                             aperture=True, lens_board=spec)
    # Geometry still comes from the border, not from the board.
    assert session.status()["calibrates_from"] == "the display's own border"
    assert session.lens_board is not None
    assert session.charuco_spec is None

    poses = [(2.0, 0.7), (9.0, -4.0), (-6.0, 5.0), (13.0, 2.0), (-11.0, -3.0),
             (4.0, 9.0), (-2.0, -8.0), (7.0, 3.5), (-9.0, 1.0), (11.0, -6.0),
             (0.5, 0.2), (-4.0, -2.0)]
    last = None
    for i, (tilt, roll) in enumerate(poses):
        _, _, shot_rig = rig_for(tilt_deg=tilt, roll_deg=roll, seed=i,
                                 sampling_ratio=0.9 + 0.02 * i)
        shot_rig.show("main")
        last = session.handle("intrinsics", jpeg(shot_rig.read()))
        assert last.verdict == "OK", last.detail
    assert "solved the lens" in last.detail
    assert session.status()["needs_intrinsics"] is False

    # And now the border route runs.
    rig.show("main")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert "nothing was asked of the cluster" in rec.detail


def test_a_lens_solve_is_judged_on_generalisation_not_on_fit():
    """rms is the wrong gate, and that is measured rather than argued.

    A set of near-identical views is fitted beautifully and is wrong
    everywhere else -- it reported 0.051 px reprojection error and left
    1.276 px of real error behind, where a good set reported 0.052 px and left
    0.093 px. Nothing in the rms told them apart. Angle spread did.
    """
    from layoutval.calibration import (
        CharucoSpec, board_points_for_intrinsics, check_intrinsics)
    from layoutval.simulator import VirtualCamera

    sensor = (2400, 1500)
    poses = [(2.0, 0.7), (9.0, -4.0), (-6.0, 5.0), (13.0, 2.0), (-11.0, -3.0),
             (4.0, 9.0), (-2.0, -8.0), (7.0, 3.5), (-9.0, 1.0), (11.0, -6.0),
             (0.5, 0.2), (-4.0, -2.0)]
    board = CharucoSpec.parse("9x6:30:22").board()
    target = cv2.cvtColor(
        board.generateImage((9 * 90, 6 * 90), marginSize=45), cv2.COLOR_GRAY2BGR)
    th, tw = target.shape[:2]
    base = min(0.75 * sensor[0] / tw, 0.75 * sensor[1] / th)

    def collect(spread):
        objs, imgs = [], []
        for i, (tilt, roll) in enumerate(poses):
            cam = VirtualCamera(
                display_size=(tw, th), sensor_size=sensor,
                sampling_ratio=base * (0.88 + 0.06 * (i % 5) * spread),
                tilt_deg=tilt * spread, roll_deg=roll * spread, seed=i,
                k1=-0.09, k2=0.02)
            try:
                o, p = board_points_for_intrinsics(cam.shoot(target), board=board)
            except RuntimeError:
                continue
            objs.append(o)
            imgs.append(p)
        return objs, imgs

    varied = check_intrinsics(*collect(1.0), sensor)
    alike = check_intrinsics(*collect(0.25), sensor)

    # The trap: the bad set's rms is no worse than the good set's.
    assert alike.rms <= varied.rms * 1.5
    # What actually separates them.
    assert varied.tilt_spread_deg > 3 * alike.tilt_spread_deg
    assert varied.complaint() is None
    assert alike.complaint() is not None
    assert "same angle" in alike.complaint()


def test_a_lens_that_does_not_generalise_is_not_adopted(tmp_path):
    """Worse than none: it beat skipping undistortion in exactly one direction."""
    from layoutval.calibration import CharucoSpec

    display, panel, rig = rig_for()
    spec = CharucoSpec(panel.squares[0], panel.squares[1], panel.square_length,
                       panel.square_length * panel.marker_ratio)
    session = CaptureSession(tmp_path, display_size=display.size,
                             aperture=True, lens_board=spec)
    # Twelve views from almost the same place: the failure rms cannot see.
    last = None
    for i in range(12):
        _, _, shot_rig = rig_for(tilt_deg=2.0 + 0.1 * i, roll_deg=0.7,
                                 seed=i, sampling_ratio=0.95)
        shot_rig.show("main")
        last = session.handle("intrinsics", jpeg(shot_rig.read()))
    assert last.verdict == "FAILED", last.detail
    assert "Not used" in last.detail
    assert session.status()["needs_intrinsics"] is True
    assert not (tmp_path / "intrinsics.json").exists()
    # The work is not thrown away.
    assert session.status()["intrinsic_views"] == 12

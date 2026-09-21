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
    content_fills_aperture,
    find_display_aperture,
    homography_from_display_aperture,
    homography_from_marked_corners,
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
    assert "display's border" in rec.detail

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
    assert "lens is not solved" in rec.detail
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
    assert "lens solved" in last.detail
    assert session.status()["needs_intrinsics"] is False

    # And now the border route runs.
    rig.show("main")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert "display's border" in rec.detail


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
    assert "same angle" in last.detail
    assert session.status()["needs_intrinsics"] is True
    assert not (tmp_path / "intrinsics.json").exists()
    # The work is not thrown away.
    assert session.status()["intrinsic_views"] == 12


def _windowed(display, outer_scale, bezel=70):
    """The HMI as a window inside a larger screen -- a laptop, in other words.

    The outer screen has the same proportions as the HMI, so the aspect test
    cannot tell them apart. That is the point: this is the case where the
    aperture route picks a plausible wrong rectangle.
    """
    inner = display.render()
    h, w = inner.shape[:2]
    ow, oh = int(w * outer_scale), int(h * outer_scale)
    outer = np.full((oh, ow, 3), bezel, np.uint8)
    x, y = (ow - w) // 2, (oh - h) // 2
    outer[y:y + h, x:x + w] = inner
    return outer


def test_a_windowed_hmi_is_answered_correctly_or_refused_never_guessed():
    """The failure that put the whole cluster in a corner of the overlay.

    An HMI running in a window inside a monitor gives two rectangles with the
    same proportions, one inside the other, and nothing in the picture says
    which is the active area. Preferring the innermost is the right principle
    and measured insufficient: across fourteen arrangements it chose correctly
    in five, and the rest were *silently* wrong -- a homography fitted to the
    outer rectangle rectifies the cluster into a corner and every element then
    fails to match, while the calibration itself reports success.

    So the property under test is not accuracy, it is that a wrong answer is
    never returned: every arrangement either measures correctly or raises.
    """
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    truth = display.render()
    probe = profile_from_reference(truth, screen="b", display_size=display.size)

    refused = correct = 0
    for scale in (1.0, 1.3, 1.6, 2.0, 2.5):
        for trim in (70, 130):
            h, w = truth.shape[:2]
            ow, oh = int(w * scale), int(h * scale)
            outer = np.full((oh, ow, 3), trim, np.uint8)
            x, y = (ow - w) // 2, (oh - h) // 2
            outer[y:y + h, x:x + w] = truth

            cam = VirtualCamera(display_size=(ow, oh), sensor_size=(2400, 1500),
                                sampling_ratio=0.9)
            live = Undistorter(Intrinsics(
                K=cam.K, dist=cam.dist,
                image_size=cam.sensor_size))(cam.shoot(outer))
            try:
                geometry = homography_from_display_aperture(
                    live, display_size=display.size)
            except RuntimeError:
                refused += 1
                continue

            rect = DisplayGeometry(
                H=geometry.H, display_size=display.size).rectify(live)
            errors = [
                float(np.hypot(m.dx, m.dy))
                for spec in probe
                for m in [measure_translation(truth, rect, spec)]
                if m.dx is not None and m.zncc and m.zncc > 0.5
            ]
            rms = (float(np.sqrt(np.mean(np.square(errors))))
                   if errors else float("inf"))
            assert rms < 0.3, (
                f"scale {scale}, trim {trim}: answered {rms:.3f} px instead of "
                "refusing -- this is the silently-wrong case"
            )
            correct += 1

    assert correct >= 2, "refusing everything would also pass the assert above"
    assert refused >= 1


def test_the_bezel_fixture_still_measures_after_the_nesting_rules():
    """The rules that refuse an ambiguous nest must not refuse a real cluster."""
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    truth = display.render()
    probe = profile_from_reference(truth, screen="b", display_size=display.size)
    for i, (tilt, roll) in enumerate([(2.0, 0.7), (6.0, -3.0), (11.0, 2.5),
                                      (-4.0, 5.0), (8.0, -8.0)]):
        _, _, rig = rig_for(tilt_deg=tilt, roll_deg=roll, seed=i)
        live = shot(rig)
        geometry = homography_from_display_aperture(
            live, display_size=display.size)
        rect = DisplayGeometry(
            H=geometry.H, display_size=display.size).rectify(live)
        errors = [float(np.hypot(m.dx, m.dy))
                  for spec in probe
                  for m in [measure_translation(truth, rect, spec)]
                  if m.dx is not None and m.zncc and m.zncc > 0.5]
        assert errors
        assert float(np.sqrt(np.mean(np.square(errors)))) < 0.3


def test_content_span_separates_right_sized_from_oversized():
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    truth = display.render()

    w, h = display.size
    exact = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    span_w, span_h, lit = content_fills_aperture(truth, exact)
    assert span_w > 0.6 and span_h > 0.6
    assert 0.0 < lit < 0.2  # a cluster is mostly dark; that is expected

    big = _windowed(display, 2.5)
    bw, bh = big.shape[1], big.shape[0]
    whole = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], np.float64)
    span_w2, span_h2, _ = content_fills_aperture(big, whole)
    assert span_w2 < 0.45 and span_h2 < 0.45


def test_the_calibration_frame_is_saved_with_the_border_drawn_on_it(tmp_path):
    """When it picks the wrong rectangle, looking at it is the fast diagnosis."""
    display, _, rig = rig_for()
    session = CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                  image_size=rig.camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), display_size=display.size)),
        display_size=display.size, aperture=True)
    rig.show("main")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail

    drawn = list(tmp_path.glob("*-aperture.jpg"))
    assert len(drawn) == 1
    assert cv2.imread(str(drawn[0])) is not None
    assert "border" in rec.detail


def test_a_loose_lens_solve_is_reported_not_refused():
    """A loose lens model still beats no lens model, so it is adopted and priced.

    Measured by adding corner-localisation noise -- what moire off a screen,
    JPEG sharpening and a soft board all amount to -- the held-out error maps
    almost linearly onto what the rig then costs:

        holdout 0.26 -> 0.113 px      holdout 1.04 -> 0.631 px
        holdout 0.52 -> 0.261 px      holdout 2.34 -> 1.296 px

    Every one of those beats skipping undistortion, which on the same lens
    costs 4.08 px. An earlier gate refused anything past 0.20 px, a figure
    taken from simulator-clean views that no phone photographing a screen can
    reach. It blocked solves that were entirely usable.
    """
    from layoutval.calibration import LensCheck

    def check(holdout, tilt=20.0):
        return LensCheck(rms=holdout * 0.9, holdout_px=holdout,
                         tilt_spread_deg=tilt, depth_spread=0.4, views=12)

    # Usable solves are adopted, and say what they will cost.
    for holdout, expected in ((0.26, 0.16), (0.52, 0.31), (1.04, 0.62)):
        c = check(holdout)
        assert c.complaint() is None, holdout
        assert c.expected_element_error_px() == pytest.approx(expected, abs=0.02)

    # Quality is a word a record can carry.
    assert check(0.05).quality() == "tight"
    assert check(0.52).quality() == "workable"
    assert check(1.04).quality() == "loose"

    # Past the point where undistorting stops being worth it, it is refused.
    assert check(3.5).complaint() is not None

    # And the failure rms cannot see is still caught, however tight it looks.
    assert check(0.05, tilt=2.5).complaint() is not None
    assert "same angle" in check(0.05, tilt=2.5).complaint()


def test_the_lens_step_stays_reachable_once_solved(tmp_path):
    """A saved lens solve must be re-shootable, not hidden behind its own success.

    `go` picks up a saved intrinsics.json, which satisfied the check and
    removed the step from the page -- so a solve made under an older, wrong
    gate could never be replaced from the phone.
    """
    display, panel, rig = rig_for()
    session = CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                  image_size=rig.camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), display_size=display.size)),
        display_size=display.size, aperture=True)

    status = session.status()
    assert status["uses_intrinsics"] is True   # the step belongs on the page
    assert status["needs_intrinsics"] is False  # but is already satisfied
    assert status["intrinsics_rms"] is not None

    # Shooting a fresh view starts a new set without dropping the working one.
    rig.show("checkerboard")
    rec = session.handle("intrinsics", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert "re-solving" in rec.detail
    assert session.status()["intrinsic_views"] == 1
    assert session.status()["has_intrinsics"] is True

    # And the border route still works while the re-shoot is half done.
    rig.show("main")
    assert session.handle("calibrate", jpeg(rig.read())).verdict == "OK"


def test_supplying_intrinsics_does_not_count_as_calibrated(tmp_path):
    """The placeholder geometry that holds the undistortion is not a mapping.

    Passing --intrinsics builds a Calibration so the lens model has somewhere
    to live, and its identity homography rectifies to a raw crop of the camera
    frame. The crop is stable between reference and validate, so every element
    matches to a few hundredths of a pixel and the whole run looks healthy --
    while the numbers are in no coordinate system at all. Skipping Calibrate
    has to fail, not succeed quietly.
    """
    from layoutval.calibration import UNSOLVED

    display, _, rig = rig_for()
    session = CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                  image_size=rig.camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), method=UNSOLVED,
                                     display_size=display.size)),
        display_size=display.size, aperture=True)

    assert session.calibration is not None      # the lens model is there
    assert session.is_calibrated is False       # the mapping is not
    assert session.status()["calibrated"] is False

    rig.show("main")
    frame = jpeg(rig.read())
    for action in ("reference", "validate"):
        rec = session.handle(action, frame)
        assert rec.verdict == "FAILED", f"{action}: {rec.detail}"
        assert "Calibrate first" in rec.detail

    # After Calibrate, both work.
    assert session.handle("calibrate", frame).verdict == "OK"
    assert session.is_calibrated is True
    assert session.handle("reference", frame).verdict == "OK"
    assert session.handle("validate", frame).verdict in ("PASS", "REVIEW")


def test_marked_corners_calibrate_without_asking_the_cluster_anything():
    """Somebody points at the display. The only route that works in any scene.

    Automatic border detection needs a clean one, and a bench is not: a real
    photograph of a laptop running the HMI contains the screen, a window inside
    it, a bezel, a laptop body and a room, several of them rectangles of about
    the right shape. The search picked a 2107x1537 region that was none of them.
    """
    display, _, rig = rig_for()
    live = shot(rig)
    truth = display.render()
    probe = profile_from_reference(truth, screen="b", display_size=display.size)

    # Where the display really is, then thrown off the way a thumb would.
    exact, _ = find_display_aperture(live, display_size=display.size)
    rng = np.random.default_rng(0)
    tapped = exact + rng.normal(0, 6.0, exact.shape)

    geometry = homography_from_marked_corners(
        live, tapped, display_size=display.size)
    rect = DisplayGeometry(
        H=geometry.H, display_size=display.size).rectify(live)
    errors = [float(np.hypot(m.dx, m.dy))
              for spec in probe
              for m in [measure_translation(truth, rect, spec)]
              if m.dx is not None and m.zncc and m.zncc > 0.5]
    assert errors, "nothing matched through the marked corners"
    # Snapped back onto the real edge, a sloppy tap still measures.
    assert geometry.method == "marked_corners"
    assert float(np.sqrt(np.mean(np.square(errors)))) < 0.5


def test_refinement_never_snaps_to_an_edge_you_did_not_point_at():
    """A boundary further away than the band that found it is a different edge.

    Opened to 5% of the display's width on a bench photograph, the fit jumped
    80 px onto the laptop's own bezel, and tap sets differing by 40 px landed
    93 px apart -- worse than not refining, since raw taps at least stay put.
    """
    display, _, rig = rig_for()
    live = shot(rig)
    exact, _ = find_display_aperture(live, display_size=display.size)

    solved = []
    for sigma in (0.0, 4.0, 9.0):
        rng = np.random.default_rng(3)
        tapped = exact + (rng.normal(0, sigma, exact.shape) if sigma else 0.0)
        g = homography_from_marked_corners(
            live, tapped, display_size=display.size)
        # Whatever it did, it did not wander off to some other edge.
        assert float(np.abs(g.corners - tapped).max()) <= 32.0
        solved.append(g.corners)

    spread = max(float(np.abs(a - b).max())
                 for i, a in enumerate(solved) for b in solved[i + 1:])
    assert spread < 25.0, f"tap error not absorbed: {spread:.1f} px apart"


def test_four_corners_are_required_and_must_enclose_something(tmp_path):
    display, _, rig = rig_for()
    session = CaptureSession(tmp_path, display_size=display.size,
                             mark_corners=True)
    assert session.status()["calibrates_from"] == "corners you mark"

    rig.show("main")
    frame = jpeg(rig.read())
    rec = session.handle("corners", frame, corners=[[0, 0], [10, 0]])
    assert rec.verdict == "FAILED"
    assert "four corners" in rec.detail

    rec = session.handle("corners", frame,
                         corners=[[0, 0], [3, 0], [3, 3], [0, 3]])
    assert rec.verdict == "FAILED"
    assert "enclose" in rec.detail

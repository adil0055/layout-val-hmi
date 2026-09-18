"""The bezel-marker route, end to end through the capture server.

This is the route for a cluster that cannot be asked to draw anything.  The
property worth testing is exactly that: after one binding frame, Calibrate has
to succeed on a frame in which the screen is showing the screen under test and
nothing has been drawn for the camera's benefit.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from layoutval.calibration import (
    Calibration,
    CharucoSpec,
    DisplayGeometry,
    Intrinsics,
    charuco_anchor,
    chessboard_display_points,
    detect_charuco,
    homography_from_charuco,
    homography_from_display_pattern,
)
from layoutval.server import CaptureSession
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera

NOMINAL = {
    "TELLTALE_BATTERY_LOW": True,
    "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True,
    "FUEL_LEVEL": 0.6,
    "SPEED": 120.0,
}


def make_rig(**camera):
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    panel = BezelPanel()
    opts = dict(display_size=panel.panel_size, sensor_size=(2400, 1500),
                sampling_ratio=0.95)
    opts.update(camera)
    return display, panel, BezelRig(display, panel, VirtualCamera(**opts))


def spec_for(panel: BezelPanel) -> CharucoSpec:
    return CharucoSpec(panel.squares[0], panel.squares[1], panel.square_length,
                       panel.square_length * panel.marker_ratio)


def jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def session_for(tmp_path, rig, panel, display):
    return CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                  image_size=rig.camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), display_size=display.size),
        ),
        pattern_size=(9, 6),
        square_px=40.0,
        pattern_origin=(60.0, 20.0),
        display_size=display.size,
        charuco=spec_for(panel),
    )


# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("16x3", (16, 3, 30.0, 22.5, "DICT_4X4_100")),
    ("5x7:30:22", (5, 7, 30.0, 22.0, "DICT_4X4_100")),
    ("9X6", (9, 6, 30.0, 22.5, "DICT_4X4_100")),
    ("9*6", (9, 6, 30.0, 22.5, "DICT_4X4_100")),
    ("9x6:25::DICT_5X5_100", (9, 6, 25.0, 18.75, "DICT_5X5_100")),
])
def test_spec_parses_what_the_flag_documents(text, expected):
    spec = CharucoSpec.parse(text)
    assert (spec.squares_x, spec.squares_y, spec.square_length,
            spec.marker_length, spec.dictionary) == expected


@pytest.mark.parametrize("text,because", [
    ("5", "COLSxROWS"),
    ("axb", "whole squares"),
    # Two rows of squares leave a single row of inner corners, and a homography
    # cannot be fitted to collinear points -- so this has to fail here rather
    # than as a confusing solver error later.
    ("5x2", "inner"),
    ("5x7:30:40", "smaller than its square"),
    ("5x7:30:22:DICT_NOT_A_THING", "COLSxROWS"),
])
def test_spec_rejects_what_cannot_work(text, because):
    with pytest.raises(ValueError) as exc:
        CharucoSpec.parse(text).board()
    assert because in str(exc.value) or "dictionary" in str(exc.value)


def test_spec_round_trips_through_its_own_dict():
    spec = CharucoSpec.parse("16x3:30:22:DICT_4X4_100")
    assert CharucoSpec(**spec.to_dict()) == spec


def test_binding_then_markers_only(tmp_path):
    """The point of the whole route: after binding, the screen is free."""
    display, panel, rig = make_rig()
    session = session_for(tmp_path, rig, panel, display)
    assert session.status()["calibrates_from"] == "bezel markers (needs binding first)"

    # Binding: the calibration pattern and the markers in one frame.
    rig.show("checkerboard")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert "bound the bezel board" in rec.detail
    assert session.charuco_anchor is not None

    # Every calibration after: the screen under test, nothing drawn for us.
    rig.show("main")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert "bezel markers alone" in rec.detail
    assert session.status()["calibrates_from"] == "bezel markers"

    # And the whole three-step run completes off that calibration.
    assert session.handle("reference", jpeg(rig.read())).verdict == "OK"
    assert session.handle("validate", jpeg(rig.read())).verdict in ("PASS", "REVIEW")


def test_markers_alone_cannot_bind(tmp_path):
    """Binding needs the active area located independently; say so."""
    display, panel, rig = make_rig()
    session = session_for(tmp_path, rig, panel, display)
    rig.show("main")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "FAILED"
    assert "has not been tied to the active area yet" in rec.detail
    assert session.charuco_anchor is None


def test_route_refuses_without_intrinsics(tmp_path):
    """Undistortion is not optional here, and the cost is lens-dependent.

    0.10 px undistorted on any lens, against 0.6-5.7 px with the distortion left
    in -- so a frame that happens to look fine on one camera says nothing about
    the next.  That is why this refuses rather than warns.
    """
    display, panel, rig = make_rig()
    session = CaptureSession(tmp_path, display_size=display.size,
                             charuco=spec_for(panel))
    rig.show("checkerboard")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "FAILED"
    assert "intrinsics" in rec.detail
    assert "calibrate-intrinsics" in rec.detail


def test_anchor_survives_a_restart(tmp_path):
    """A sticker on a bezel outlives the process that measured it."""
    display, panel, rig = make_rig()
    first = session_for(tmp_path, rig, panel, display)
    rig.show("checkerboard")
    assert first.handle("calibrate", jpeg(rig.read())).verdict == "OK"

    revived = session_for(tmp_path, rig, panel, display)
    assert revived.charuco_anchor is not None
    assert revived.charuco_bind_corners
    rig.show("main")
    assert revived.handle("calibrate", jpeg(rig.read())).verdict == "OK"


def test_anchor_for_a_different_board_is_not_reused(tmp_path):
    """The anchor is in board units, so the wrong board would rescale silently."""
    display, panel, rig = make_rig()
    first = session_for(tmp_path, rig, panel, display)
    rig.show("checkerboard")
    assert first.handle("calibrate", jpeg(rig.read())).verdict == "OK"

    other = CaptureSession(tmp_path, display_size=display.size,
                           charuco=CharucoSpec(9, 6, 25.0, 18.0))
    assert other.charuco_anchor is None

    saved = json.loads((tmp_path / "charuco-anchor.json").read_text())
    assert saved["spec"] == spec_for(panel).to_dict()


def test_rebind_needs_a_board_to_rebind(tmp_path):
    display, panel, rig = make_rig()
    session = CaptureSession(tmp_path, display_size=display.size)
    rig.show("checkerboard")
    rec = session.handle("rebind", jpeg(rig.read()))
    assert rec.verdict == "FAILED"
    assert "no bezel board to re-bind" in rec.detail


def test_moving_the_camera_is_reported_not_hidden(tmp_path):
    """Reuse across a camera move stays plausible and gets worse; say which."""
    display, panel, rig = make_rig()
    session = session_for(tmp_path, rig, panel, display)
    rig.show("checkerboard")
    assert session.handle("calibrate", jpeg(rig.read())).verdict == "OK"

    # Same bezel, same board, a different place to stand.
    _, _, moved = make_rig(tilt_deg=12.0, roll_deg=-7.0, seed=3)
    moved.display.state.update(NOMINAL)
    moved.show("main")
    rec = session.handle("calibrate", jpeg(moved.read()))
    assert rec.verdict == "OK", rec.detail
    assert "from where it was bound" in rec.detail
    assert "Re-bind from here" in rec.detail


def test_rebinding_restores_the_tighter_figure(tmp_path):
    """Re-binding in the pose in use is the fix, so it has to actually work."""
    display, panel, rig = make_rig()
    session = session_for(tmp_path, rig, panel, display)
    rig.show("checkerboard")
    assert session.handle("calibrate", jpeg(rig.read())).verdict == "OK"
    before = dict(session.charuco_bind_corners)

    _, panel2, moved = make_rig(tilt_deg=12.0, roll_deg=-7.0, seed=3)
    moved.display.state.update(NOMINAL)
    moved.show("checkerboard")
    rec = session.handle("rebind", jpeg(moved.read()))
    assert rec.verdict == "OK", rec.detail
    assert "bound the bezel board" in rec.detail
    assert session.charuco_bind_corners != before

    moved.show("main")
    rec = session.handle("calibrate", jpeg(moved.read()))
    assert rec.verdict == "OK", rec.detail
    assert "from where it was bound" in rec.detail
    assert "Re-bind from here" not in rec.detail


def test_printed_board_round_trips_through_the_detector(tmp_path):
    """A board the CLI draws has to be one the detector reads back.

    Both ends come from the same spec string, so this catches the way that can
    go wrong: a margin too tight for the detector to find the outer squares, or
    a generated image whose proportions do not match the spec that will be used
    to read it.
    """
    from layoutval.cli import build_parser

    out = tmp_path / "board.png"
    args = build_parser().parse_args(
        ["charuco-board", "16x3:30:22", "--out", str(out), "--dpi", "60"])
    assert args.func(args) == 0

    printed = cv2.imread(str(out))
    assert printed is not None
    corners, ids = detect_charuco(printed, CharucoSpec.parse("16x3:30:22").board())
    # 16x3 squares leave 15x2 = 30 inner corners.
    assert len(corners) == 30
    assert sorted(int(i) for i in ids) == list(range(30))


def test_printed_size_follows_the_spec_not_the_dpi(tmp_path):
    """The spec's lengths are the physical size; dpi only sets the pixels."""
    from layoutval.cli import build_parser

    sizes = {}
    for dpi in (60, 120):
        out = tmp_path / f"board-{dpi}.png"
        args = build_parser().parse_args(
            ["charuco-board", "9x6:20:15", "--out", str(out), "--dpi", str(dpi),
             "--margin-mm", "0"])
        assert args.func(args) == 0
        h, w = cv2.imread(str(out)).shape[:2]
        sizes[dpi] = (w, h)
        # 9x6 squares of 20 mm is 180 x 120 mm, so the aspect is fixed at 1.5.
        assert w / h == pytest.approx(1.5, abs=0.01)
    assert sizes[120][0] == pytest.approx(2 * sizes[60][0], abs=2)


def test_intrinsics_can_be_collected_from_the_phone(tmp_path):
    """The lens solve is reachable from the bench, not only from a shell.

    This is the whole reason the route no longer refuses at startup: the step
    that used to mean shooting a set of photographs and moving files by hand is
    the one most likely to stop somebody before they start.
    """
    display, panel, rig = make_rig()
    session = CaptureSession(tmp_path, display_size=display.size,
                             charuco=spec_for(panel))
    assert session.status()["needs_intrinsics"] is True

    # Calibrate refuses, and says what to do about it.
    rig.show("checkerboard")
    assert session.handle("calibrate", jpeg(rig.read())).verdict == "FAILED"

    poses = [(2.0, 0.7), (9.0, -4.0), (-6.0, 5.0), (13.0, 2.0), (-11.0, -3.0),
             (4.0, 9.0), (-2.0, -8.0), (7.0, 3.5), (-9.0, 1.0), (11.0, -6.0),
             (0.5, 0.2), (-4.0, -2.0)]
    last = None
    for i, (tilt, roll) in enumerate(poses):
        _, _, shot = make_rig(tilt_deg=tilt, roll_deg=roll, seed=i,
                              sampling_ratio=0.9 + 0.02 * i)
        shot.show("main")
        last = session.handle("intrinsics", jpeg(shot.read()))
        assert last.verdict == "OK", last.detail

    assert "solved the lens" in last.detail
    assert (tmp_path / "intrinsics.json").exists()
    assert session.calibration.intrinsics is not None
    assert session.status()["needs_intrinsics"] is False

    # And the route it was blocking now runs.
    rig.show("checkerboard")
    rec = session.handle("calibrate", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert "bound the bezel board" in rec.detail


def test_intrinsic_views_must_share_a_frame_size(tmp_path):
    """Intrinsics are in pixels, so mixing sizes fits neither."""
    display, panel, rig = make_rig()
    session = CaptureSession(tmp_path, display_size=display.size,
                             charuco=spec_for(panel))
    rig.show("main")
    assert session.handle("intrinsics", jpeg(rig.read())).verdict == "OK"

    _, _, other = make_rig(sensor_size=(1600, 1000))
    other.show("main")
    rec = session.handle("intrinsics", jpeg(other.read()))
    assert rec.verdict == "FAILED"
    assert "same camera at the same resolution" in rec.detail


def test_a_view_without_the_board_is_not_counted(tmp_path):
    display, panel, rig = make_rig()
    session = CaptureSession(tmp_path, display_size=display.size,
                             charuco=spec_for(panel))
    blank = np.full((900, 1600, 3), 40, np.uint8)
    rec = session.handle("intrinsics", jpeg(blank))
    assert rec.verdict == "FAILED"
    assert "Not counted" in rec.detail
    assert session.status()["intrinsic_views"] == 0

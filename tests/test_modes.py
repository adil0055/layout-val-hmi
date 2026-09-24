"""Switching how the display is located, live, from the phone.

Three modes: corners found automatically and confirmed, corners tapped by hand,
and the cluster's own chessboard. They map into different display spaces -- the
screen's corners for the first two, the board's canvas for the last -- which is
the part most likely to go quietly wrong when switching between them.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np
import pytest

from layoutval.calibration import (
    UNSOLVED,
    Calibration,
    DisplayGeometry,
    Intrinsics,
    chessboard_display_points,
)
from layoutval.server import CaptureSession, serve
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera

NOMINAL = {
    "TELLTALE_BATTERY_LOW": True, "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True, "FUEL_LEVEL": 0.6, "SPEED": 120.0,
}
PATTERN, SQUARE_PX, ORIGIN = (9, 6), 40.0, (60.0, 20.0)


@pytest.fixture()
def rig():
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    return BezelRig(display, BezelPanel(), VirtualCamera(
        display_size=BezelPanel().panel_size, sensor_size=(2400, 1500),
        sampling_ratio=0.95, tilt_deg=4.0, roll_deg=-2.0, seed=1))


def make_session(tmp_path, rig, *, board=True, mode="auto"):
    display = rig.display
    return CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=rig.camera.K, dist=rig.camera.dist,
                                  image_size=rig.camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), method=UNSOLVED,
                                     display_size=display.size)),
        display_size=display.size,
        pattern_size=PATTERN,
        display_points=(chessboard_display_points(PATTERN, SQUARE_PX, ORIGIN)
                        if board else None),
        board_display_size=display.size if board else None,
        mark_corners=True,
        calib_mode=mode,
    )


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def upload(url, action, frame, corners=None):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    fields = [("action", action.encode())]
    if corners is not None:
        fields.append(("corners", json.dumps(corners).encode()))
    for name, value in fields:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                   f'name="{name}"\r\n\r\n'.encode() + value + b"\r\n")
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
               f'filename="c.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
               + jpeg(frame) + f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=body.getvalue(), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def set_mode(base, token, mode):
    req = urllib.request.Request(
        f"{base}/mode?t={token}", data=json.dumps({"mode": mode}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------


def test_status_offers_the_modes_and_says_which_is_on(tmp_path, rig):
    session = make_session(tmp_path, rig)
    s = session.status()
    assert s["mode"] == "auto"
    assert s["modes"] == {"manual": True, "auto": True, "chessboard": True}
    assert s["calibrates_from"] == "corners found automatically"


def test_chessboard_is_refused_without_a_board(tmp_path, rig):
    """Guessing the board is how a smaller grid solves at the wrong scale."""
    session = make_session(tmp_path, rig, board=False)
    assert session.status()["modes"]["chessboard"] is False
    with pytest.raises(ValueError, match="--board"):
        session.set_mode("chessboard")
    assert session.calib_mode == "auto"


def test_each_mode_maps_into_its_own_display_space(tmp_path, rig):
    session = make_session(tmp_path, rig)
    session.corner_display_size = (1920, 1200)
    session.board_display_size = (1790, 870)
    session.set_mode("chessboard")
    assert session.display_size == (1790, 870)
    session.set_mode("manual")
    assert session.display_size == (1920, 1200)


def test_the_chessboard_needs_no_lens_solve_and_the_corners_do(tmp_path, rig):
    session = make_session(tmp_path, rig)
    session.calibration = None
    session.set_mode("chessboard")
    assert session.status()["needs_intrinsics"] is False
    session.set_mode("auto")
    assert session.status()["needs_intrinsics"] is True


def test_propose_changes_nothing_until_confirmed(tmp_path, rig):
    session = make_session(tmp_path, rig)
    rig.show("main")
    rec = session.handle("propose", jpeg(rig.read()))
    assert rec.verdict == "OK", rec.detail
    assert len(rec.proposal["corners"]) == 4
    assert not session.is_calibrated
    assert session.history == []


def test_auto_mode_end_to_end_over_http(tmp_path, rig):
    """What the phone does: propose, confirm the dots, reference, validate."""
    session = make_session(tmp_path, rig)
    srv = serve(session, host="127.0.0.1", port=0, quiet=True)
    try:
        host, port = srv.server_address
        base = f"http://{host}:{port}"
        url = f"{base}/upload?t={srv.token}"
        rig.show("main")
        frame = rig.read()
        proposed = upload(url, "propose", frame)
        assert proposed["verdict"] == "OK", proposed
        corners = proposed["proposal"]["corners"]
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in corners)

        calibrated = upload(url, "corners", frame, corners)
        assert calibrated["verdict"] == "OK", calibrated
        assert upload(url, "reference", frame)["verdict"] == "OK"
        assert upload(url, "validate", frame)["verdict"] == "PASS"
    finally:
        srv.shutdown()
        srv.server_close()


def test_switching_to_the_chessboard_over_http(tmp_path, rig):
    session = make_session(tmp_path, rig)
    srv = serve(session, host="127.0.0.1", port=0, quiet=True)
    try:
        host, port = srv.server_address
        base = f"http://{host}:{port}"
        status = set_mode(base, srv.token, "chessboard")
        assert status["mode"] == "chessboard"
        assert status["calibrates_from"] == "the cluster's chessboard"
        rig.show("checkerboard")
        board_frame = rig.read()
        rig.show("main")
        result = upload(f"{base}/upload?t={srv.token}", "calibrate", board_frame)
        assert result["verdict"] == "OK", result

        with pytest.raises(urllib.error.HTTPError) as err:
            set_mode(base, srv.token, "sideways")
        assert err.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_mode_switching_needs_the_token(tmp_path, rig):
    session = make_session(tmp_path, rig)
    srv = serve(session, host="127.0.0.1", port=0, quiet=True)
    try:
        host, port = srv.server_address
        with pytest.raises(urllib.error.HTTPError) as err:
            set_mode(f"http://{host}:{port}", "wrongtoken", "manual")
        assert err.value.code == 403
        assert session.calib_mode == "auto"
    finally:
        srv.shutdown()
        srv.server_close()


def test_switching_mode_starts_calibration_over_but_keeps_the_lens(tmp_path, rig):
    """A new mode is a new mapping into a different display space.

    Left calibrated, the page saw no reference and moved on to Reference, and
    the chessboard photograph meant to calibrate became the reference instead.
    """
    session = make_session(tmp_path, rig)
    rig.show("main")
    frame = jpeg(rig.read())
    proposal = session.handle("propose", frame).proposal
    h, w = rig.read().shape[:2]
    corners = [[x * w, y * h] for x, y in proposal["corners"]]
    assert session.handle("corners", frame, corners=corners).verdict == "OK"
    assert session.handle("reference", frame).verdict == "OK"
    lens = session.calibration.intrinsics

    session.set_mode("chessboard")
    assert not session.is_calibrated
    assert session.reference is None
    assert session.calibration.intrinsics is lens

    # Choosing the same mode again is not a switch and costs nothing.
    rig.show("checkerboard")
    assert session.handle("calibrate", jpeg(rig.read())).verdict == "OK"
    session.set_mode("chessboard")
    assert session.is_calibrated


def test_accepting_a_proposal_unchanged_snaps_to_the_panel_edge(tmp_path, rig):
    """Proposed on the undistorted frame, handed back in raw pixels, then refined.

    Proposed on the raw photograph instead, lens distortion bowed the edges and
    the corners met 29-34 px off -- enough that the snap-to-edge step refused
    them and the calibration went through unrefined.
    """
    from layoutval.calibration import Undistorter, find_display_aperture

    session = make_session(tmp_path, rig)
    rig.show("main")
    raw = rig.read()
    proposal = session.handle("propose", jpeg(raw)).proposal
    h, w = raw.shape[:2]
    corners = [[x * w, y * h] for x, y in proposal["corners"]]
    session.handle("corners", jpeg(raw), corners=corners)
    assert session.calibration.geometry.method == "marked_corners"

    undistorted = Undistorter(session.calibration.intrinsics)(raw)
    border, _ = find_display_aperture(undistorted, display_size=rig.display.size)
    dw, dh = rig.display.size
    solved = cv2.perspectiveTransform(
        np.array([[[-0.5, -0.5]], [[dw - 0.5, -0.5]], [[dw - 0.5, dh - 0.5]],
                  [[-0.5, dh - 0.5]]], np.float64),
        session.calibration.geometry.H).reshape(-1, 2)
    assert float(np.abs(solved - border).max()) < 0.5

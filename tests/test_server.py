"""The phone-capture path, end to end against the simulator.

A "phone" here is the virtual camera photographing the synthetic cluster, and
the frames go over a real socket to a real server, so the multipart parsing,
the token check, the orientation handling and the three actions are all
exercised the way a phone would exercise them.
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

from layoutval.server import (
    ACTIONS,
    MAX_UPLOAD_BYTES,
    CaptureSession,
    apply_orientation,
    decode_upload,
    exif_orientation,
    lan_address,
    serve,
)
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera

from conftest import NOMINAL, VALUES, build_bench

PATTERN = (9, 6)
SQUARE_PX = 40
PATTERN_ORIGIN = (60, 20)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def post(url: str, action: str, image: np.ndarray, *, as_jpeg: bool = True) -> dict:
    """Upload a frame the way the capture page does."""
    ext = ".jpg" if as_jpeg else ".png"
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return post_bytes(url, action, buf.tobytes(), "capture" + ext)


def post_bytes(url: str, action: str, payload: bytes, filename: str) -> dict:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="action"\r\n\r\n')
    body.write(action.encode() + b"\r\n")
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
    )
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(payload + b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


@pytest.fixture(scope="module")
def rig():
    """A phone-like camera looking at the cluster, uncalibrated."""
    display = ClusterDisplay()
    camera = VirtualCamera(display_size=display.size)
    r = SimulatedRig(display, camera)
    display.state.update(NOMINAL)
    return r


@pytest.fixture()
def server(tmp_path, rig):
    session = CaptureSession(
        tmp_path / "captures",
        pattern_size=PATTERN,
        square_px=SQUARE_PX,
        pattern_origin=PATTERN_ORIGIN,
        display_size=rig.display.size,
        values=VALUES,
    )
    srv = serve(session, host="127.0.0.1", port=0, quiet=True)
    yield srv
    srv.shutdown()
    srv.server_close()


def shoot(rig, screen: str = "main") -> np.ndarray:
    rig.show(screen)
    frame = rig.read()
    rig.show("main")
    return frame


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------


def test_orientation_of_a_frame_without_exif_is_upright():
    ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    assert ok
    assert exif_orientation(buf.tobytes()) == 1


def test_orientation_transforms_round_trip():
    """A phone records which way up it was held instead of rotating the pixels."""
    img = np.zeros((20, 40, 3), np.uint8)
    img[0:5, 0:10] = 255  # a corner mark, so a rotation is detectable
    assert np.array_equal(apply_orientation(img, 1), img)
    assert apply_orientation(img, 6).shape[:2] == (40, 20)
    assert apply_orientation(img, 8).shape[:2] == (40, 20)
    assert np.array_equal(apply_orientation(apply_orientation(img, 3), 3), img)
    # An unknown value must pass the frame through rather than mangle it.
    assert np.array_equal(apply_orientation(img, 99), img)


def test_decode_upload_rejects_rubbish():
    with pytest.raises(ValueError, match="did not decode"):
        decode_upload(b"this is not an image")


def test_lan_address_is_a_v4_address():
    parts = lan_address().split(".")
    assert len(parts) == 4 and all(p.isdigit() for p in parts)


# --------------------------------------------------------------------------
# access control
# --------------------------------------------------------------------------


def test_the_page_needs_the_token(server):
    host, port = server.server_address
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"http://{host}:{port}/", timeout=10)
    assert exc.value.code == 401

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"http://{host}:{port}/status?t=wrong", timeout=10)
    assert exc.value.code == 403


def test_uploads_need_the_token(server, rig):
    host, port = server.server_address
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(f"http://{host}:{port}/upload?t=wrong", "calibrate", shoot(rig))
    assert exc.value.code == 403


def test_the_page_is_served_with_the_token(server):
    host, port = server.server_address
    with urllib.request.urlopen(f"{server.url}", timeout=10) as response:
        body = response.read().decode()
    assert "Cluster capture" in body
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "no-store" in response.headers["Cache-Control"]


def test_a_traversal_out_of_the_capture_directory_is_refused(server):
    host, port = server.server_address
    for name in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"http://{host}:{port}/shot/{name}?t={server.token}", timeout=10
            )
        assert exc.value.code == 404


def test_an_oversized_upload_is_refused(server):
    host, port = server.server_address
    request = urllib.request.Request(
        f"http://{host}:{port}/upload?t={server.token}",
        data=b"x",
        headers={"Content-Type": "multipart/form-data; boundary=x",
                 "Content-Length": str(MAX_UPLOAD_BYTES + 1)},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=10)
    assert exc.value.code == 413


def test_a_post_that_is_not_an_image_is_refused_politely(server):
    """A bench tool reports and keeps serving; it does not fall over."""
    host, port = server.server_address
    with pytest.raises(urllib.error.HTTPError) as exc:
        post_bytes(f"http://{host}:{port}/upload?t={server.token}",
                   "calibrate", b"not an image at all", "x.jpg")
    assert exc.value.code == 400
    # Still alive.
    assert get(f"http://{host}:{port}/status?t={server.token}")["captures"] == 0


# --------------------------------------------------------------------------
# the three actions
# --------------------------------------------------------------------------


def test_status_before_anything_has_happened(server):
    status = get(f"http://{server.server_address[0]}:{server.server_address[1]}"
                 f"/status?t={server.token}")
    assert status["calibrated"] is False
    assert status["has_reference"] is False
    assert status["captures"] == 0


def test_validate_refuses_before_calibration(server, rig):
    host, port = server.server_address
    url = f"http://{host}:{port}/upload?t={server.token}"
    result = post(url, "validate", shoot(rig))
    assert result["verdict"] == "FAILED"
    assert "calibrate" in result["detail"]


def test_reference_refuses_before_calibration(server, rig):
    host, port = server.server_address
    url = f"http://{host}:{port}/upload?t={server.token}"
    result = post(url, "reference", shoot(rig))
    assert result["verdict"] == "FAILED"
    assert "calibrate" in result["detail"]


def test_calibration_says_so_when_the_board_is_not_there(server, rig):
    """The usual cause is the wrong screen, and the message has to say that."""
    host, port = server.server_address
    url = f"http://{host}:{port}/upload?t={server.token}"
    result = post(url, "calibrate", shoot(rig, "main"))
    assert result["verdict"] == "FAILED"
    assert "board" in result["detail"] or "pattern not found" in result["detail"]


def test_calibrate_then_reference_then_validate(server, rig, tmp_path):
    """The whole loop, over the wire, the way a phone walks it."""
    host, port = server.server_address
    url = f"http://{host}:{port}/upload?t={server.token}"

    result = post(url, "calibrate", shoot(rig, "checkerboard"))
    assert result["verdict"] == "OK", result["detail"]
    assert "sampling ratio" in result["detail"]
    status = get(f"http://{host}:{port}/status?t={server.token}")
    assert status["calibrated"] is True
    assert status["sampling_ratio"] > 1.0

    result = post(url, "reference", shoot(rig))
    assert result["verdict"] == "OK", result["detail"]
    assert get(f"http://{host}:{port}/status?t={server.token}")["has_reference"] is True

    # No profile was loaded, so there is no inventory to measure against.
    result = post(url, "validate", shoot(rig))
    assert result["verdict"] == "FAILED"
    assert "profile" in result["detail"]

    assert (server.session.out_dir / "calibration.json").is_file()


def test_recalibrating_drops_the_reference(server, rig):
    """The geometry moved, so anything measured against the old one is void."""
    host, port = server.server_address
    url = f"http://{host}:{port}/upload?t={server.token}"
    assert post(url, "calibrate", shoot(rig, "checkerboard"))["verdict"] == "OK"
    assert post(url, "reference", shoot(rig))["verdict"] == "OK"
    assert get(f"http://{host}:{port}/status?t={server.token}")["has_reference"]
    assert post(url, "calibrate", shoot(rig, "checkerboard"))["verdict"] == "OK"
    assert not get(f"http://{host}:{port}/status?t={server.token}")["has_reference"]


def test_a_captured_fault_is_measured_and_reported(tmp_path):
    """A defect injected into the display comes back through the phone path."""
    bench = build_bench()
    bench.reset()

    session = CaptureSession(
        tmp_path / "captures",
        pattern_size=PATTERN,
        square_px=SQUARE_PX,
        pattern_origin=PATTERN_ORIGIN,
        display_size=bench.display.size,
        fixed_camera=True,
        values=VALUES,
    )
    srv = serve(session, host="127.0.0.1", port=0, quiet=True)
    try:
        host, port = srv.server_address
        url = f"http://{host}:{port}/upload?t={srv.token}"

        bench.rig.show("checkerboard")
        assert post(url, "calibrate", bench.rig.read())["verdict"] == "OK"
        bench.rig.show("main")

        # Build a profile against the reference the server just took.
        assert post(url, "reference", bench.rig.read())["verdict"] == "OK"
        reference = session.reference
        assert reference is not None

        import json as _json
        from pathlib import Path as _Path

        from layoutval.profile import import_design_tree
        from layoutval.types import Tolerance

        export = _json.loads(
            (_Path(__file__).resolve().parents[1] / "examples" / "design_export.json").read_text()
        )
        profile = import_design_tree(
            export, screen="main", display_size=bench.display.size,
            defaults=Tolerance(tol_warn=1.5, tol_fail=2.5, spec_tolerance_px=1.0),
        )
        # Named explicitly: the raw design export has no travel model for the
        # fuel bar and no pivot for the needle, and neither is what this test is
        # about. The elements kept are the plain static ones.
        keep = {"TELLTALE_BATTERY_LOW", "TELLTALE_OIL_PRESSURE", "TELLTALE_ABS",
                "LABEL_SPEED_UNITS", "GAUGE_PLATE"}
        profile.elements = [s for s in profile if s.id in keep]
        session.profile = profile

        clean = post(url, "validate", bench.rig.read())
        assert clean["verdict"] in ("PASS", "REVIEW"), clean

        bench.display.offsets["TELLTALE_ABS"] = (6.0, 0.0)
        faulty = post(url, "validate", bench.rig.read())
        assert faulty["verdict"] == "FAIL", faulty
        moved = [r for r in faulty["rows"] if r["id"] == "TELLTALE_ABS"]
        assert moved, faulty["rows"]
        assert "6." in moved[0]["detail"] or "5." in moved[0]["detail"], moved

        overlay = session.out_dir / faulty["overlay"]
        assert overlay.is_file()
    finally:
        srv.shutdown()
        srv.server_close()


def test_actions_are_a_closed_set():
    assert set(ACTIONS) == {"calibrate", "reference", "validate"}

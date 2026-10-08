"""SAM 3 mode: the screen found from a text prompt, by a model on another computer.

The real service needs a GPU and Meta's checkpoint, so these run the real
``sam3_server/sam3_server.py`` HTTP service with a stand-in segmenter that
answers with the true screen's mask -- everything between the phone and the
model is exercised: the upload with its prompt, the service and its token, the
mask's journey back, the shape fitted to it, the corners, the reference's
screen and the verdicts.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import threading
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from layoutval import sam3client
from layoutval.calibration import UNSOLVED, Calibration, DisplayGeometry, Intrinsics, Undistorter
from layoutval.server import KEEP_WHOLE_PHOTO, PAGE, CaptureSession, serve
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("PIL")


def _load_service():
    spec = importlib.util.spec_from_file_location(
        "sam3_server", ROOT / "sam3_server" / "sam3_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSam3:
    """Answers like SAM 3 would, with the screen ``truth`` draws, plus a decoy."""

    device = "test"

    def __init__(self, truth):
        self.truth = truth
        self.prompts: list[str] = []

    def segment(self, image, prompt, threshold, limit):
        self.prompts.append(prompt)
        if prompt == "a giraffe":
            return []
        w, h = image.size
        screen = self.truth(w, h)
        decoy = np.zeros((h, w), bool)
        decoy[: h // 12, : w // 12] = True
        return [{"mask": screen, "score": 0.92, "box": [0, 0, w, h]},
                {"mask": decoy, "score": 0.45, "box": [0, 0, w / 12, h / 12]}][:limit]


@pytest.fixture()
def service():
    """Start the real SAM 3 service with a stand-in model; yields a starter."""
    module = _load_service()
    started = []

    def start(truth, token="s3cret"):
        model = FakeSam3(truth)
        srv = module.Sam3Server(("127.0.0.1", 0), model, token=token)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        host, port = srv.server_address
        return f"http://{host}:{port}", model

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def _to_undistorted(camera, pts):
    """Ideal-camera pixels (the simulator's H_true) to the undistorted frame's,
    as the session undistorts: keeping the whole photograph."""
    new_k = Undistorter(Intrinsics(K=camera.K, dist=camera.dist, image_size=camera.sensor_size,
                                   alpha=KEEP_WHOLE_PHOTO)).new_K
    m = new_k @ np.linalg.inv(camera.K)
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), m).reshape(-1, 2)


def _truth(camera, outline_in_camera):
    """The screen as a mask of whatever size the service was sent."""
    pts = _to_undistorted(camera, outline_in_camera)
    sw, sh = camera.sensor_size

    def draw(w, h):
        mask = np.zeros((h, w), np.uint8)
        scaled = pts * [w / sw, h / sh]
        cv2.fillPoly(mask, [np.round(scaled * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
        return mask > 0
    return draw


def _session(tmp_path, camera, display_size, url, token="s3cret"):
    return CaptureSession(
        tmp_path,
        calibration=Calibration(
            intrinsics=Intrinsics(K=camera.K, dist=camera.dist, image_size=camera.sensor_size),
            geometry=DisplayGeometry(H=np.eye(3), method=UNSOLVED, display_size=display_size)),
        display_size=display_size,
        mark_corners=True,
        calib_mode="sam3",
        sam3_url=url,
        sam3_token=token,
    )


def _jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def _pixels(proposal, frame):
    h, w = frame.shape[:2]
    return [[x * w, y * h] for x, y in proposal["corners"]]


# -- a rectangular screen ---------------------------------------------------


@pytest.fixture()
def rig():
    display = ClusterDisplay()
    display.state.update({"TELLTALE_BATTERY_LOW": True, "TELLTALE_OIL_PRESSURE": True,
                          "TELLTALE_ABS": True, "FUEL_LEVEL": 0.6, "SPEED": 120.0})
    return BezelRig(display, BezelPanel(), VirtualCamera(
        display_size=BezelPanel().panel_size, sensor_size=(2400, 1500),
        sampling_ratio=0.95, tilt_deg=4.0, roll_deg=-2.0, seed=1))


def _opening(rig):
    """The screen's opening in the bezel, in the ideal camera's pixels."""
    x, y, w, h = rig.panel.display_rect
    quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64) - 0.5
    return cv2.perspectiveTransform(quad.reshape(-1, 1, 2), rig.camera.H_true).reshape(-1, 2)


def _rect_truth(rig):
    return _truth(rig.camera, _opening(rig))


def test_a_rectangular_screen_is_found_from_the_prompt_and_snapped(tmp_path, rig, service):
    url, model = service(_rect_truth(rig))
    session = _session(tmp_path, rig.camera, rig.display.size, url)
    rig.show("main")
    raw = rig.read()

    rec = session.handle("propose", _jpeg(raw), prompt="instrument cluster")
    assert rec.verdict == "OK", rec.detail
    assert model.prompts == ["instrument cluster"]
    assert rec.proposal["shape"] == "rect" and rec.proposal["confident"]
    assert len(rec.proposal["outline"]) >= 4
    assert not session.is_calibrated                       # nothing until confirmed

    done = session.handle("corners", _jpeg(raw), corners=_pixels(rec.proposal, raw))
    assert done.verdict == "OK", done.detail
    assert "SAM 3" in done.detail and session.calibration.geometry.method == "marked_corners"

    # Snapped to the panel's edge, as tapped corners are.
    dw, dh = rig.display.size
    quad = np.array([[[-0.5, -0.5]], [[dw - 0.5, -0.5]], [[dw - 0.5, dh - 0.5]], [[-0.5, dh - 0.5]]])
    solved = cv2.perspectiveTransform(quad, session.calibration.geometry.H).reshape(-1, 2)
    assert float(np.abs(solved - _to_undistorted(rig.camera, _opening(rig))).max()) < 1.0

    ref = session.handle("reference", _jpeg(raw))
    assert ref.verdict == "OK" and "SAM 3" not in ref.detail
    assert model.prompts[-1] == "instrument cluster"       # asked again for the same thing
    # A rectangle fills its corners: there is nothing to trim.
    assert session.screen_mask is None or session.screen_mask.mean() > 0.97
    assert session.handle("validate", _jpeg(rig.read())).verdict == "PASS"


def test_the_default_prompt_is_used_when_none_is_typed(tmp_path, rig, service):
    url, model = service(_rect_truth(rig))
    session = _session(tmp_path, rig.camera, rig.display.size, url)
    rig.show("main")
    assert session.handle("propose", _jpeg(rig.read())).verdict == "OK"
    assert model.prompts == [sam3client.DEFAULT_PROMPT]


# -- a round screen ---------------------------------------------------------


ROUND = 1200
RADIUS = 600          # inscribed: a round module's framebuffer is the square round it


def _round_screen(moved=(0, 0)):
    """A round cluster's framebuffer: a dial with ticks and readouts in a circle."""
    img = np.zeros((ROUND, ROUND, 3), np.uint8)
    c = ROUND // 2
    cv2.circle(img, (c, c), RADIUS, (38, 34, 30), -1, cv2.LINE_AA)
    for k in range(0, 270, 18):
        a = np.radians(135 + k)
        p = (int(c + 470 * np.cos(a)), int(c + 470 * np.sin(a)))
        q = (int(c + 520 * np.cos(a)), int(c + 520 * np.sin(a)))
        cv2.line(img, p, q, (230, 230, 230), 6, cv2.LINE_AA)
    cv2.putText(img, "120", (c - 150, c + 40), cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                (240, 240, 240), 10, cv2.LINE_AA)
    cv2.putText(img, "km/h", (c - 80, c + 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (200, 200, 200), 4, cv2.LINE_AA)
    dx, dy = moved
    cv2.rectangle(img, (c - 220 + dx, c + 250 + dy), (c - 120 + dx, c + 300 + dy), (60, 200, 90), -1)
    cv2.rectangle(img, (c + 120, c + 250), (c + 220, c + 300), (60, 140, 230), -1)
    cv2.circle(img, (c, c - 260), 40, (40, 60, 230), -1, cv2.LINE_AA)
    return img


@pytest.fixture()
def round_camera():
    return VirtualCamera(display_size=(ROUND, ROUND), sensor_size=(2000, 1500),
                         sampling_ratio=1.0, tilt_deg=3.0, roll_deg=0.0, seed=3,
                         pwm_amplitude=0.0)


def _round_truth(camera):
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    disc = np.column_stack([ROUND / 2 + RADIUS * np.cos(t), ROUND / 2 + RADIUS * np.sin(t)])
    in_camera = cv2.perspectiveTransform(disc.reshape(-1, 1, 2), camera.H_true)
    return _truth(camera, in_camera)


def test_a_round_screen_is_found_measured_and_kept_to_its_circle(tmp_path, round_camera, service):
    url, model = service(_round_truth(round_camera))
    session = _session(tmp_path, round_camera, (ROUND, ROUND), url)
    raw = round_camera.shoot(_round_screen())

    rec = session.handle("propose", _jpeg(raw), prompt="round instrument cluster")
    assert rec.verdict == "OK", rec.detail
    assert rec.proposal["shape"] == "round"

    done = session.handle("corners", _jpeg(raw), corners=_pixels(rec.proposal, raw))
    assert done.verdict == "OK", done.detail
    # Not snapped: a circle has no straight edge to snap the dots to.
    assert session.calibration.geometry.method == "marked_corners(unrefined)"

    # The centre of the dial lands on the centre of the framebuffer, give or
    # take what a circle cannot say: seen 3 degrees off square, the centre of
    # its ellipse is about r^2 tan(tilt) / distance -- 10 px here -- from the
    # image of its centre. The same mapping serves reference and validation,
    # so it is the same in both and cancels.
    centre = cv2.perspectiveTransform(np.array([[[ROUND / 2, ROUND / 2]]]),
                                      session.calibration.geometry.H).reshape(2)
    true = _to_undistorted(round_camera, cv2.perspectiveTransform(
        np.array([[[ROUND / 2, ROUND / 2]]]), round_camera.H_true)).reshape(2)
    assert float(np.hypot(*(centre - true))) < 12.0

    assert session.handle("reference", _jpeg(raw)).verdict == "OK"
    # Only the circle is screen: its framebuffer's corners are bezel.
    share = float(session.screen_mask.mean())
    assert 0.70 < share < np.pi / 4
    assert session.profile is not None and len(session.profile) >= 3

    same = session.handle("validate", _jpeg(round_camera.shoot(_round_screen())))
    assert same.verdict == "PASS", same.detail
    # A plain block 4 px off. (Judged night or day on the whole framebuffer,
    # the black corners round the circle read this night theme as a day one,
    # the block was taken for a reflection, and it passed even 20 px off.)
    moved = session.handle("validate", _jpeg(round_camera.shoot(_round_screen(moved=(4, 0)))))
    assert moved.verdict == "FAIL", moved.detail
    failed = [e for e in moved.report["elements"] if e["verdict"] == "FAIL"]
    assert len(failed) == 1 and 3.0 < failed[0]["measurement"]["abs_delta"] < 5.0


# -- when it cannot help ----------------------------------------------------


def test_nothing_found_says_so_and_changes_nothing(tmp_path, rig, service):
    url, _ = service(_rect_truth(rig))
    session = _session(tmp_path, rig.camera, rig.display.size, url)
    rig.show("main")
    rec = session.handle("propose", _jpeg(rig.read()), prompt="a giraffe")
    assert rec.verdict == "FAILED" and "a giraffe" in rec.detail
    assert rec.proposal is None and not session.is_calibrated


def test_a_wrong_token_is_refused(tmp_path, rig, service):
    url, model = service(_rect_truth(rig), token="right")
    session = _session(tmp_path, rig.camera, rig.display.size, url, token="wrong")
    rig.show("main")
    rec = session.handle("propose", _jpeg(rig.read()))
    assert rec.verdict == "FAILED" and "token" in rec.detail
    assert model.prompts == []


def test_an_unreachable_sam3_computer_does_not_break_the_session(tmp_path, rig):
    session = _session(tmp_path, rig.camera, rig.display.size, "http://127.0.0.1:9")
    rig.show("main")
    raw = rig.read()
    rec = session.handle("propose", _jpeg(raw))
    assert rec.verdict == "FAILED" and "could not reach" in rec.detail
    # The dots can still be placed by hand, and the reference still works.
    from layoutval.displayfind import propose_display_corners
    undistorter = Undistorter(session.calibration.intrinsics)
    found = propose_display_corners(undistorter(raw))
    from layoutval.server import _distort_points
    corners = _distort_points(found.corners, session.calibration.intrinsics, undistorter.new_K)
    assert session.handle("corners", _jpeg(raw), corners=corners.tolist()).verdict == "OK"
    ref = session.handle("reference", _jpeg(raw))
    assert ref.verdict == "OK" and "could not reach" in ref.detail


def test_sam3_is_only_offered_with_a_sam3_computer(tmp_path, rig):
    session = CaptureSession(tmp_path, display_size=rig.display.size,
                             mark_corners=True, calib_mode="auto")
    assert "sam3" not in session.status()["modes"]
    assert session.status()["sam3_prompt"] is None
    with pytest.raises(ValueError, match="--sam3-url"):
        session.set_mode("sam3")

    offered = CaptureSession(tmp_path, display_size=rig.display.size, mark_corners=True,
                             calib_mode="auto", sam3_url="http://gpu:8765",
                             sam3_prompt="round cluster")
    assert offered.status()["modes"]["sam3"] is True
    assert offered.status()["sam3_prompt"] == "round cluster"
    offered.set_mode("sam3")
    assert offered.mark_corners and offered.status()["calibrates_from"] == "the screen SAM 3 finds"


# -- the phone --------------------------------------------------------------


def _upload(url, fields, frame):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for name, value in fields:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                   f'name="{name}"\r\n\r\n'.encode() + value.encode() + b"\r\n")
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
               f'filename="c.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
               + _jpeg(frame) + f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url, data=body.getvalue(), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_the_phone_sends_its_prompt_and_confirms_the_dots(tmp_path, rig, service):
    url, model = service(_rect_truth(rig))
    session = _session(tmp_path, rig.camera, rig.display.size, url)
    srv = serve(session, host="127.0.0.1", port=0, quiet=True)
    try:
        host, port = srv.server_address
        upload = f"http://{host}:{port}/upload?t={srv.token}"
        rig.show("main")
        frame = rig.read()
        proposed = _upload(upload, [("action", "propose"), ("prompt", "  car dashboard  screen ")], frame)
        assert proposed["verdict"] == "OK", proposed
        assert model.prompts == ["car dashboard screen"]
        assert proposed["proposal"]["prompt"] == "car dashboard screen"
        corners = proposed["proposal"]["corners"]
        done = _upload(upload, [("action", "corners"), ("corners", json.dumps(corners))], frame)
        assert done["verdict"] == "OK" and "SAM 3" in done["detail"]
        assert _upload(upload, [("action", "reference")], frame)["verdict"] == "OK"
        assert _upload(upload, [("action", "validate")], frame)["verdict"] == "PASS"
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_page_has_the_sam3_mode_and_an_optional_prompt():
    assert 'data-m="sam3"' in PAGE
    assert 'id="sam3prompt"' in PAGE and "optional" in PAGE
    assert 'body.append("prompt"' in PAGE


# -- the line the design keeps ------------------------------------------------


def test_the_client_only_talks_http():
    tree = ast.parse((ROOT / "src" / "layoutval" / "sam3client.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(n.split(".")[0] in ("torch", "sam3", "transformers", "PIL") for n in names)


def test_the_service_checks_what_it_is_sent(service):
    url, model = service(lambda w, h: np.ones((h, w), bool))

    def post(path, data, token="s3cret"):
        req = urllib.request.Request(url + path, data=data, method="POST",
                                     headers={"X-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    img = _jpeg(np.zeros((40, 60, 3), np.uint8))
    assert post("/segment?prompt=screen", img, token="nope")[0] == 403
    assert post("/segment?prompt=", img)[0] == 400
    assert post("/segment?prompt=screen", b"not a picture")[0] == 400
    status, body = post("/segment?prompt=screen&threshold=0&max=1", img)
    assert status == 200 and len(body["results"]) == 1 and (body["width"], body["height"]) == (60, 40)
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        assert json.loads(r.read())["ok"] is True

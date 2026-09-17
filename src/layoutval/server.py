"""Capture from a phone on the same network, measure on the machine running this.

The rig this package assumes is a camera bolted in front of a cluster.  Getting
to that point takes a mount, a lens and an afternoon.  Long before then it is
useful to be able to point a phone at the screen and have the answer come back,
and this is that: a small HTTP server on the laptop, a page the phone opens, and
the whole pipeline behind it.

Three actions, which are the three things a rig needs in order:

``calibrate``
    The cluster is showing a chessboard (``layoutval calibrate-geometry``'s
    pattern, or the HMI's own calibration screen).  Solves display-to-camera
    from the uploaded frame and stores it.  Nothing else works until this has
    been done once.
``reference``
    The cluster is showing the screen under test, correct.  Rectifies the
    uploaded frame and keeps it as the golden reference.
``validate``
    Measures the uploaded frame against that reference and answers.

**A phone in your hand is not a fixed camera, and this cannot pretend
otherwise.**  Every number this package produces rests on the camera not
moving.  A hand-held frame is at a different pose from the one that was
calibrated, and the difference is a perspective change rather than a nudge, so
the server re-solves the pose of each frame against the reference before
measuring (:class:`DriftTracker` in homography mode) and reports how far it had
to go.  That buys back a usable measurement from a hand-held shot, at a cost
that is stated rather than hidden: a correction that re-solves the whole pose
also absorbs a fault in which *everything* moved together.  Per-element faults
survive it, because the rest of the frame dominates the fit; a whole-layout
shift does not.  Clamp the phone and the trade goes away -- pass
``--fixed-camera`` and the pose is checked but not re-solved.

Security, such as it is: this binds to the local network and accepts uploads, so
it is a bench tool and not something to leave running.  Every URL carries a
token minted at startup, uploads are capped and have to decode as an image, and
nothing from an upload is ever executed or used as a path.  It speaks plain HTTP
because a phone camera over HTTPS needs a certificate the phone trusts, which is
not a thing to inflict on a test rig; the capture page therefore uses the
file-upload control that hands off to the phone's own camera app, which needs no
secure context.
"""

from __future__ import annotations

import json
import secrets
import socket
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from layoutval.calibration import (
    Calibration,
    DriftTracker,
    Undistorter,
    chessboard_display_points,
    homography_from_display_pattern,
)
from layoutval.pipeline import Pipeline, PipelineOptions
from layoutval.profile import LayoutProfile
from layoutval.report import annotate
from layoutval.types import RunReport, Verdict

#: Biggest upload accepted.  A phone photograph is a few megabytes; anything an
#: order of magnitude past that is not a photograph.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

ACTIONS = ("calibrate", "reference", "validate")


# --------------------------------------------------------------------------
# JPEG orientation
# --------------------------------------------------------------------------


def exif_orientation(data: bytes) -> int:
    """The EXIF orientation tag of a JPEG, or 1 when there is not one.

    Phones record which way up they were held rather than rotating the pixels,
    and :func:`cv2.imdecode` ignores that.  A frame that came in on its side
    would calibrate and measure perfectly happily and be wrong about everything,
    so it is worth the fifty lines.
    """
    if not data.startswith(b"\xff\xd8"):
        return 1
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return 1
        marker, size = data[i + 1], struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker == 0xE1 and data[i + 4 : i + 10] == b"Exif\x00\x00":
            return _orientation_from_tiff(data[i + 10 : i + 2 + size])
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of scan: no EXIF after this
            return 1
        i += 2 + size
    return 1


def _orientation_from_tiff(tiff: bytes) -> int:
    if len(tiff) < 8:
        return 1
    endian = "<" if tiff[:2] == b"II" else ">" if tiff[:2] == b"MM" else None
    if endian is None:
        return 1
    try:
        offset = struct.unpack(endian + "I", tiff[4:8])[0]
        count = struct.unpack(endian + "H", tiff[offset : offset + 2])[0]
        for n in range(count):
            entry = offset + 2 + n * 12
            tag = struct.unpack(endian + "H", tiff[entry : entry + 2])[0]
            if tag == 0x0112:
                value = struct.unpack(endian + "H", tiff[entry + 8 : entry + 10])[0]
                return value if 1 <= value <= 8 else 1
    except (struct.error, IndexError):
        return 1
    return 1


def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Put the pixels the way up the phone was holding them."""
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def decode_upload(data: bytes) -> np.ndarray:
    """Bytes off the wire to a BGR frame, the right way up."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("the upload did not decode as an image")
    return apply_orientation(img, exif_orientation(data))


def lan_address() -> str:
    """This machine's address on the local network.

    Found by opening a UDP socket towards an address that is never contacted --
    the kernel picks the interface it would route through, which is the one the
    phone can reach.  ``gethostname`` would as often as not return 127.0.0.1.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1: reserved, never routed
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------


@dataclass
class CaptureRecord:
    """One upload and what came of it."""

    name: str
    action: str
    when: str
    verdict: str = ""
    detail: str = ""
    report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action": self.action,
            "when": self.when,
            "verdict": self.verdict,
            "detail": self.detail,
        }


class CaptureSession:
    """What the server knows between requests, behind one lock.

    Uploads arrive on whatever thread the server hands them to, and calibrating
    or re-referencing replaces state the next request reads, so every mutation
    goes through the lock.  Without it, two phones (or one impatient phone)
    could measure against a half-replaced reference and the result would be
    plausible rather than obviously wrong.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        profile: LayoutProfile | None = None,
        calibration: Calibration | None = None,
        pattern_size: tuple[int, int] = (9, 6),
        square_px: float = 100.0,
        pattern_origin: tuple[float, float] = (0.0, 0.0),
        display_size: tuple[int, int] | None = None,
        fixed_camera: bool = False,
        drift_alarm_px: float = 2.0,
        values: dict[str, float] | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.calibration = calibration
        self.pattern_size = pattern_size
        self.square_px = square_px
        self.pattern_origin = pattern_origin
        self.fixed_camera = fixed_camera
        self.drift_alarm_px = drift_alarm_px
        self.values = values or {}
        self.display_size = display_size or (
            calibration.geometry.display_size if calibration else (1920, 720)
        )
        self.reference: np.ndarray | None = None
        self.reference_camera: np.ndarray | None = None
        self.history: list[CaptureRecord] = []

        if profile is not None and profile.reference_path:
            try:
                self.reference = profile.reference()
            except RuntimeError:
                self.reference = None

    # -- helpers ------------------------------------------------------------

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.calibration and self.calibration.intrinsics:
            return Undistorter(self.calibration.intrinsics)(frame)
        return frame

    def status(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibration is not None,
            "has_reference": self.reference is not None,
            "has_profile": self.profile is not None,
            "elements": len(self.profile) if self.profile else 0,
            "display_size": list(self.display_size),
            "fixed_camera": self.fixed_camera,
            "sampling_ratio": (
                round(self.calibration.geometry.sampling_ratio(), 3)
                if self.calibration else None
            ),
            "captures": len(self.history),
        }

    def _store(self, name: str, img: np.ndarray) -> str:
        cv2.imwrite(str(self.out_dir / name), img)
        return name

    def _record(self, rec: CaptureRecord) -> CaptureRecord:
        self.history.insert(0, rec)
        del self.history[40:]
        return rec

    # -- actions ------------------------------------------------------------

    def handle(self, action: str, data: bytes) -> CaptureRecord:
        frame = decode_upload(data)
        stamp = datetime.now().strftime("%H%M%S")
        base = f"{stamp}-{action}"
        with self.lock:
            if action == "calibrate":
                return self._record(self._calibrate(frame, base))
            if action == "reference":
                return self._record(self._reference(frame, base))
            if action == "validate":
                return self._record(self._validate(frame, base))
        raise ValueError(f"unknown action {action!r}")

    def _calibrate(self, frame: np.ndarray, base: str) -> CaptureRecord:
        rec = CaptureRecord(name=f"{base}.jpg", action="calibrate",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        undistorted = self._undistort(frame)
        points = chessboard_display_points(
            self.pattern_size, self.square_px, self.pattern_origin
        )
        try:
            geometry = homography_from_display_pattern(
                undistorted, self.pattern_size, points, display_size=self.display_size
            )
        except RuntimeError as exc:
            rec.verdict = "FAILED"
            rec.detail = (
                f"{exc} Check the whole board is in frame, square on, and that "
                f"--pattern matches what the cluster is drawing "
                f"({self.pattern_size[0]}x{self.pattern_size[1]} inner corners)."
            )
            return rec

        self.calibration = Calibration(
            intrinsics=self.calibration.intrinsics if self.calibration else None,
            geometry=geometry,
            rig={"source": "phone capture"},
            drift_alarm_px=self.drift_alarm_px,
        )
        self.calibration.save(self.out_dir / "calibration.json")
        ratio = geometry.sampling_ratio()
        rec.verdict = "OK"
        rec.detail = (
            f"solved from {len(points)} corners, fit residual "
            f"{geometry.residual_px:.3f} camera px, sampling ratio {ratio:.2f} "
            "camera px per display px"
        )
        if ratio < 2.0:
            rec.detail += (
                ". Below 2 you cannot reliably resolve a one-display-pixel "
                "shift -- move closer or zoom in before trusting a tolerance."
            )
        # The geometry has moved, so anything measured against the old one is
        # meaningless.  Drop it rather than let it be compared across.
        self.reference = None
        self.reference_camera = None
        return rec

    def _reference(self, frame: np.ndarray, base: str) -> CaptureRecord:
        rec = CaptureRecord(name=f"{base}.jpg", action="reference",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        if self.calibration is None:
            rec.verdict = "FAILED"
            rec.detail = "calibrate first: there is no display-to-camera mapping yet"
            return rec
        undistorted = self._undistort(frame)
        self.reference_camera = undistorted
        self.reference = self.calibration.geometry.rectify(undistorted)
        self._store(f"{base}-rectified.png", self.reference)
        if self.profile is not None:
            self.profile.reference_path = f"{base}-rectified.png"
            self.profile.root = self.out_dir
        rec.verdict = "OK"
        rec.detail = (
            f"rectified to {self.reference.shape[1]}x{self.reference.shape[0]} "
            "display px and kept as the reference"
        )
        return rec

    def _validate(self, frame: np.ndarray, base: str) -> CaptureRecord:
        rec = CaptureRecord(name=f"{base}.jpg", action="validate",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        if self.calibration is None:
            rec.verdict = "FAILED"
            rec.detail = "calibrate first"
            return rec
        if self.reference is None:
            rec.verdict = "FAILED"
            rec.detail = "take a reference first: there is nothing to measure against"
            return rec
        if self.profile is None:
            rec.verdict = "FAILED"
            rec.detail = "no layout profile loaded, so there is no inventory to measure"
            return rec

        undistorted = self._undistort(frame)
        report = RunReport(screen=self.profile.screen, theme=self.profile.theme)

        tracker = None
        if self.reference_camera is not None:
            h, w = undistorted.shape[:2]
            tracker = DriftTracker(
                self.reference_camera,
                (0, 0, w, h),
                alarm_px=self.drift_alarm_px,
                motion=(cv2.MOTION_EUCLIDEAN if self.fixed_camera
                        else cv2.MOTION_HOMOGRAPHY),
            )
        # Stages 4-6 only: there is one frame rather than a live source, and
        # the pose below is resolved here, so the pipeline is handed an already
        # rectified frame.
        pipeline = Pipeline(
            self.calibration,
            self.profile,
            options=PipelineOptions(settle=False, frames_per_measurement=1),
        )

        H = self.calibration.geometry.H
        if tracker is not None:
            est = tracker.measure(undistorted)
            report.metadata["pose_shift_px"] = round(est.magnitude_px, 2)
            if not est.converged:
                rec.verdict = "FAILED"
                rec.detail = (
                    "could not line this frame up with the reference. The camera "
                    "has moved too far, or it is not looking at the same screen."
                )
                return rec
            H = tracker.corrected_homography(self.calibration.geometry, est)
            if self.fixed_camera and est.exceeds:
                report.flag(
                    "camera_moved", severity="review",
                    shift_px=round(est.magnitude_px, 2),
                    detail="the camera has moved since the reference was taken",
                )
            elif not self.fixed_camera:
                report.flag(
                    "pose_resolved", severity="review",
                    shift_px=round(est.magnitude_px, 2),
                    detail=(
                        "hand-held: this frame's pose was re-solved against the "
                        "reference. A fault in which every element moved together "
                        "would be absorbed by that and not reported."
                    ),
                )

        live = self.calibration.geometry.rectify(undistorted, H)
        self._store(f"{base}-rectified.png", live)
        report = pipeline.measure_frame(
            live, values=self.values, reference=self.reference, report=report
        )

        overlay = annotate(live, report, self.profile, values=self.values)
        self._store(f"{base}-overlay.png", overlay)
        (self.out_dir / f"{base}.json").write_text(json.dumps(report.to_dict(), indent=2))

        rec.verdict = report.verdict.value
        rec.report = report.to_dict()
        failures = [r for r in report.results if r.verdict is not Verdict.PASS]
        rec.detail = (
            f"{len(report.results)} elements measured, {len(failures)} not passing"
        )
        return rec


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>Cluster capture</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0F1518;color:#E3EAE8;font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
     padding:16px 16px calc(16px + env(safe-area-inset-bottom));-webkit-text-size-adjust:100%}
h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:#8B9AA3;font-size:13px;margin:0 0 16px}
.card{background:#171F23;border:1px solid #2B373D;border-radius:10px;padding:14px;margin-bottom:12px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.step{background:#1E272C;border:1px solid #2B373D;border-radius:9px;padding:10px 8px;text-align:center;
      cursor:pointer;user-select:none;-webkit-tap-highlight-color:transparent}
.step.on{background:#1D5B8A;border-color:#3E8FD0}
.step.done{border-color:#2F7D57}
.step b{display:block;font-size:14px}
.step span{display:block;font-size:11px;color:#9DADB5;margin-top:2px}
.step.on span{color:#CFE4F5}
label.shoot{display:block;background:#2F7D57;border-radius:10px;padding:18px;text-align:center;
            font-size:17px;font-weight:600;cursor:pointer;-webkit-tap-highlight-color:transparent}
label.shoot:active{background:#276848}
label.shoot.busy{background:#3A4650;color:#9DADB5}
input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
.hint{color:#8B9AA3;font-size:12.5px;margin-top:10px}
.verdict{font-size:22px;font-weight:700;letter-spacing:.02em}
.PASS,.OK{color:#4FB584}.FAIL,.FAILED{color:#E8695C}.REVIEW{color:#DCA83F}
.detail{color:#C2CED6;font-size:13.5px;margin-top:4px}
img.shot{width:100%;border-radius:8px;border:1px solid #2B373D;margin-top:10px;display:block}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
td{padding:5px 4px;border-bottom:1px solid #222C31}
td:last-child{text-align:right;color:#9DADB5;font-variant-numeric:tabular-nums}
.warn{background:#2A2314;border:1px solid #6B5417;color:#DCA83F;border-radius:8px;
      padding:9px 11px;font-size:12.5px;margin-top:10px}
.stat{display:flex;justify-content:space-between;font-size:13px;color:#9DADB5;padding:3px 0}
.stat b{color:#E3EAE8;font-weight:600;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<h1>Cluster capture</h1>
<p class="sub">Point at the cluster and shoot. Measuring happens on the laptop.</p>

<div class="steps" id="steps">
  <div class="step" data-a="calibrate"><b>1 Calibrate</b><span>chessboard up</span></div>
  <div class="step" data-a="reference"><b>2 Reference</b><span>correct screen</span></div>
  <div class="step" data-a="validate"><b>3 Validate</b><span>screen under test</span></div>
</div>

<div class="card">
  <label class="shoot" id="shootLabel" for="shot">Take photo</label>
  <input id="shot" type="file" accept="image/*" capture="environment">
  <div class="hint" id="hint"></div>
</div>

<div class="card" id="result" style="display:none"></div>

<div class="card">
  <div class="stat"><span>Calibrated</span><b id="s-cal">—</b></div>
  <div class="stat"><span>Reference</span><b id="s-ref">—</b></div>
  <div class="stat"><span>Elements</span><b id="s-el">—</b></div>
  <div class="stat"><span>Sampling ratio</span><b id="s-sr">—</b></div>
  <div class="stat"><span>Captures</span><b id="s-n">—</b></div>
</div>

<script>
const TOKEN = new URLSearchParams(location.search).get("t") || "";
let action = "calibrate";
const $ = id => document.getElementById(id);

document.querySelectorAll(".step").forEach(el => {
  el.onclick = () => { action = el.dataset.a; paintSteps(); };
});
function paintSteps() {
  document.querySelectorAll(".step").forEach(el =>
    el.classList.toggle("on", el.dataset.a === action));
  $("hint").textContent = {
    calibrate: "Put the cluster on its chessboard pattern first. Fill the frame with it, square on.",
    reference: "Show the screen under test, correct. This becomes what later shots are compared against.",
    validate: "Show the same screen with whatever you are testing. Keep the phone where it was."
  }[action];
}

async function refresh() {
  try {
    const r = await fetch(`/status?t=${TOKEN}`);
    const s = await r.json();
    $("s-cal").textContent = s.calibrated ? "yes" : "no";
    $("s-ref").textContent = s.has_reference ? "yes" : "no";
    $("s-el").textContent = s.has_profile ? s.elements : "no profile";
    $("s-sr").textContent = s.sampling_ratio ? s.sampling_ratio.toFixed(2) : "—";
    $("s-n").textContent = s.captures;
    document.querySelector('[data-a="calibrate"]').classList.toggle("done", s.calibrated);
    document.querySelector('[data-a="reference"]').classList.toggle("done", s.has_reference);
    if (s.calibrated && !s.has_reference && action === "calibrate") { action = "reference"; paintSteps(); }
  } catch (e) { /* the laptop went away; the next poll will say so */ }
}

$("shot").onchange = async ev => {
  const file = ev.target.files[0];
  if (!file) return;
  const label = $("shootLabel");
  label.textContent = "Measuring…";
  label.classList.add("busy");
  const body = new FormData();
  body.append("action", action);
  body.append("image", file, file.name || "capture.jpg");
  try {
    const r = await fetch(`/upload?t=${TOKEN}`, { method: "POST", body });
    show(await r.json());
  } catch (e) {
    show({ verdict: "FAILED", detail: "could not reach the laptop: " + e });
  }
  label.textContent = "Take photo";
  label.classList.remove("busy");
  ev.target.value = "";
  refresh();
};

function show(res) {
  const box = $("result");
  box.style.display = "block";
  let html = `<div class="verdict ${res.verdict}">${res.verdict || "?"}</div>`;
  if (res.detail) html += `<div class="detail">${esc(res.detail)}</div>`;
  (res.flags || []).forEach(f => { html += `<div class="warn">${esc(f)}</div>`; });
  if (res.rows && res.rows.length) {
    html += "<table>" + res.rows.map(r =>
      `<tr><td class="${r.verdict}">${esc(r.id)}</td><td>${esc(r.detail)}</td></tr>`).join("") + "</table>";
  }
  if (res.overlay) html += `<img class="shot" src="/shot/${res.overlay}?t=${TOKEN}" alt="annotated capture">`;
  box.innerHTML = html;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
const esc = s => String(s).replace(/[<>&"]/g, c =>
  ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));

paintSteps();
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "layoutval"
    sys_version = ""

    # -- plumbing -----------------------------------------------------------
    @property
    def session(self) -> CaptureSession:
        return self.server.session  # type: ignore[attr-defined]

    def _authorised(self, query: dict[str, list[str]]) -> bool:
        given = (query.get("t") or [""])[0]
        return secrets.compare_digest(given, self.server.token)  # type: ignore[attr-defined]

    def _send(self, code: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A capture page has no business being framed or sniffed, and the
        # results are per-run, so nothing here should be cached.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: HTTPStatus, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        print(f"  {self.address_string()} {fmt % args}")

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/favicon.ico":
            # Asked for unprompted, and with no token; answering 403 puts an
            # error in the phone's console that looks like a broken page.
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        if url.path == "/" and not query.get("t"):
            # No token: say so rather than 404, because the usual cause is a
            # bookmarked URL from an earlier run, and the token changes.
            self._send(HTTPStatus.UNAUTHORIZED,
                       b"Open the link the laptop printed; it carries a one-run token.",
                       "text/plain; charset=utf-8")
            return
        if not self._authorised(query):
            self._send(HTTPStatus.FORBIDDEN, b"bad token", "text/plain; charset=utf-8")
            return
        if url.path == "/":
            self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/status":
            self._json(HTTPStatus.OK, self.session.status())
        elif url.path.startswith("/shot/"):
            self._serve_shot(url.path[len("/shot/"):])
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

    def _serve_shot(self, name: str) -> None:
        # Names come from our own records, never from the request, but resolve
        # and check anyway: a path from a URL is exactly how a directory
        # traversal gets in.
        out = self.session.out_dir.resolve()
        target = (out / name).resolve()
        if out not in target.parents or not target.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        kind = "image/png" if target.suffix == ".png" else "image/jpeg"
        self._send(HTTPStatus.OK, target.read_bytes(), kind)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._authorised(parse_qs(url.query)):
            self._send(HTTPStatus.FORBIDDEN, b"bad token", "text/plain; charset=utf-8")
            return
        if url.path != "/upload":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                       {"verdict": "FAILED",
                        "detail": f"upload must be 1..{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"})
            return

        try:
            action, image = _parse_multipart(
                self.rfile.read(length), self.headers.get("Content-Type", "")
            )
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"verdict": "FAILED", "detail": str(exc)})
            return
        if action not in ACTIONS:
            self._json(HTTPStatus.BAD_REQUEST,
                       {"verdict": "FAILED", "detail": f"unknown action {action!r}"})
            return

        started = time.monotonic()
        try:
            record = self.session.handle(action, image)
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"verdict": "FAILED", "detail": str(exc)})
            return
        except Exception as exc:  # a bench tool: report, do not take the server down
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"verdict": "FAILED", "detail": f"{type(exc).__name__}: {exc}"})
            return

        payload = _payload(record)
        payload["seconds"] = round(time.monotonic() - started, 2)
        if not self.server.quiet:  # type: ignore[attr-defined]
            print(f"  {action}: {record.verdict} — {record.detail}")
        self._json(HTTPStatus.OK, payload)


def _payload(record: CaptureRecord) -> dict[str, Any]:
    """What the phone needs to show, and no more."""
    out: dict[str, Any] = {
        "verdict": record.verdict,
        "detail": record.detail,
        "action": record.action,
    }
    report = record.report
    if not report:
        return out
    out["flags"] = [f.get("detail") or f.get("flag", "") for f in report.get("flags", [])]
    rows = []
    for element in report.get("elements", []):
        if element["verdict"] == "PASS":
            continue
        m = element["measurement"]
        delta = m.get("abs_delta")
        reason = element["reason"] or ""
        # No distance for an element that was not drawn: there is nothing for it
        # to be a distance from, and a number there reads as a near miss.
        show_delta = delta is not None and not m.get("element_absent")
        rows.append({
            "id": element["element_id"],
            "verdict": element["verdict"],
            "detail": f"{delta:.2f} px  {reason}".strip() if show_delta else reason,
        })
    out["rows"] = rows[:24]
    summary = report.get("summary", {})
    if summary:
        out["detail"] = (f"{summary.get('PASS', 0)} pass, "
                         f"{summary.get('REVIEW', 0)} review, "
                         f"{summary.get('FAIL', 0)} fail")
    out["overlay"] = record.name.rsplit(".", 1)[0] + "-overlay.png"
    return out


def _parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Pull the action and the image out of a browser form post.

    Through :mod:`email` rather than by hand: multipart boundaries inside
    binary image data are a classic way to get a hand-rolled parser to hand
    back the wrong bytes.
    """
    if "multipart/form-data" not in content_type:
        raise ValueError("expected a multipart form post")
    message = BytesParser(policy=default_policy).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    action, image = "", b""
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if name == "action":
            action = payload.decode("utf-8", "replace").strip()
        elif name == "image":
            image = payload
    if not image:
        raise ValueError("no image in the upload")
    return action, image


class CaptureServer(ThreadingHTTPServer):
    """The bench server.  One session, many phones, no state on the phone."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, session: CaptureSession, host: str, port: int,
                 *, token: str = "", quiet: bool = False) -> None:
        super().__init__((host, port), _Handler)
        self.session = session
        self.token = token or secrets.token_urlsafe(9)
        self.quiet = quiet

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        if host in ("0.0.0.0", "::", ""):
            host = lan_address()
        return f"http://{host}:{port}/?t={self.token}"


def serve(session: CaptureSession, *, host: str = "0.0.0.0", port: int = 8000,
          quiet: bool = False) -> CaptureServer:
    """Start the server on a background thread and hand it back."""
    server = CaptureServer(session, host, port, quiet=quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

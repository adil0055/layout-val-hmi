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

from layoutval.autoprofile import profile_from_reference
from layoutval.displayfind import propose_display_corners
from layoutval.calibration import (
    Calibration,
    CharucoSpec,
    DisplayGeometry,
    DriftTracker,
    Undistorter,
    UNSOLVED,
    charuco_anchor,
    check_intrinsics,
    draw_aperture,
    find_display_aperture,
    chessboard_display_points,
    detect_charuco,
    homography_from_charuco,
    homography_from_display_aperture,
    homography_from_display_pattern,
    homography_from_marked_corners,
    homography_from_screen_content,
    board_points_for_intrinsics,
    solve_intrinsics,
)
from layoutval.pipeline import Pipeline, PipelineOptions
from layoutval.profile import LayoutProfile
from layoutval.report import annotate
from layoutval.types import RunReport, Verdict

#: Biggest upload accepted.  A phone photograph is a few megabytes; anything an
#: order of magnitude past that is not a photograph.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

ACTIONS = ("intrinsics", "calibrate", "propose", "corners", "rebind",
           "reference", "validate")

#: How the display is located, as the phone offers them. ``manual``: tap the
#: four corners. ``auto``: the corners are proposed and you confirm or drag
#: them. ``chessboard``: the cluster draws its calibration pattern.
CALIBRATION_MODES = ("manual", "auto", "chessboard")

#: Views wanted before the intrinsics solve is attempted.  Eight is the same
#: floor the offline command uses: a calibration solved from three
#: near-identical views is worse than none, because it looks fine.
INTRINSIC_VIEWS_WANTED = 12

#: How far the board is allowed to move in the frame before the anchor
#: learned in the bind frame stops being the anchor for this one.  Measured,
#: not guessed: with the anchor re-bound in the pose it is used in, bezel
#: markers land at 0.09 px median against 0.05 px for a board on the screen;
#: reused across a deliberate camera move they go to 0.25-0.32 px median and
#: 0.56-0.94 px at p95.  That is the cost this threshold exists to flag.
ANCHOR_POSE_TOLERANCE_PX = 25.0

#: Alphabet the URL token is drawn from.
#:
#: No ``0``/``o``, ``1``/``l``/``i``, and no capitals. Somebody is going to read
#: this off a laptop and type it into a phone, and every one of those pairs is a
#: way for that to fail in a manner that looks like the server is broken. The
#: token is also compared case-insensitively, because a phone keyboard will
#: capitalise the first character given half a chance.
TOKEN_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"
TOKEN_LENGTH = 10


def make_token(length: int = TOKEN_LENGTH) -> str:
    """A token that survives being read aloud and typed in."""
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(length))


def token_matches(given: str, expected: str) -> bool:
    """Constant-time, case-insensitive, and safe on rubbish input."""
    if not isinstance(given, str) or not given.isascii():
        return False
    return secrets.compare_digest(given.strip().lower(), expected.strip().lower())


# --------------------------------------------------------------------------
# getting the link onto the phone
# --------------------------------------------------------------------------


def qr_matrix(text: str, quiet: int = 4) -> np.ndarray:
    """QR code for ``text`` as a boolean matrix, True where a module is dark.

    Through OpenCV, which is already a dependency and both encodes and decodes,
    so ``tests/test_server.py`` can check a rendered code by reading it back
    rather than by trusting it.
    """
    encoded = cv2.QRCodeEncoder.create().encode(text)
    # OpenCV writes 0 for a dark module and ships two modules of quiet zone;
    # the specification asks for four, and a phone camera notices the
    # difference on a busy terminal.
    dark = np.asarray(encoded) == 0
    return np.pad(dark, quiet, constant_values=False)


def qr_terminal(text: str, *, quiet: int = 4) -> str:
    """The QR as text, one character per module, two module rows per line.

    Rendered with half blocks so the code comes out roughly square in a
    terminal where characters are about twice as tall as they are wide, and
    with explicit foreground and background colours so it scans on a light
    terminal theme as well as a dark one -- a code that inverts with the user's
    colour scheme is one a camera will refuse about half the time.
    """
    m = qr_matrix(text, quiet=quiet)
    if m.shape[0] % 2:
        m = np.vstack([m, np.zeros((1, m.shape[1]), bool)])
    black_fg, white_fg = "\x1b[38;5;0m", "\x1b[38;5;15m"
    black_bg, white_bg = "\x1b[48;5;0m", "\x1b[48;5;15m"
    lines = []
    for y in range(0, m.shape[0], 2):
        row = []
        for x in range(m.shape[1]):
            top, bottom = m[y, x], m[y + 1, x]
            row.append((black_fg if top else white_fg)
                       + (black_bg if bottom else white_bg) + "\u2580")
        lines.append("".join(row) + "\x1b[0m")
    return "\n".join(lines)


def qr_width(text: str, quiet: int = 4) -> int:
    """How many terminal columns :func:`qr_terminal` will need."""
    return int(qr_matrix(text, quiet=quiet).shape[1])


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
    #: Proposed corners, for the phone to show as draggable dots.
    proposal: dict[str, Any] | None = None

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
        display_points: np.ndarray | None = None,
        display_size: tuple[int, int] | None = None,
        fixed_camera: bool = False,
        drift_alarm_px: float = 2.0,
        values: dict[str, float] | None = None,
        auto_profile: bool = True,
        render: np.ndarray | None = None,
        charuco: CharucoSpec | None = None,
        lens_board: CharucoSpec | None = None,
        aperture: bool = False,
        mark_corners: bool = False,
        display_inset_px: tuple[float, float] = (0.0, 0.0),
        calib_mode: str = "",
        board_display_size: tuple[int, int] | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.calibration = calibration
        self.pattern_size = pattern_size
        self.square_px = square_px
        self.pattern_origin = pattern_origin
        self.display_points = (
            None if display_points is None
            else np.asarray(display_points, dtype=np.float64).reshape(-1, 2)
        )
        self.fixed_camera = fixed_camera
        self.drift_alarm_px = drift_alarm_px
        self.values = values or {}
        self.auto_profile = auto_profile
        self._profile_is_auto = False
        #: The framebuffer, in display coordinates. When present, Calibrate
        #: matches the screen's own artwork instead of needing a pattern drawn
        #: for it, so the cluster never has to leave the screen under test.
        self.render = render
        #: Calibrate from the display's own physical border. This is the only
        #: route that asks the cluster for nothing at all -- not a pattern, not
        #: a framebuffer, and not the one binding frame the bezel markers need.
        self.aperture = aperture
        #: Somebody points at the four corners of the display, once. The only
        #: route that works in any scene, because the part that is hard to
        #: automate -- which rectangle in the picture is the display -- is the
        #: part a person does instantly.
        self.mark_corners = mark_corners
        self.display_inset_px = display_inset_px
        #: A board stuck to the bezel, when there is one.  This is the route for
        #: a cluster you cannot ask to draw anything: the markers live outside
        #: the active area, so the screen under test can stay on the screen
        #: under test.  It costs one binding frame, once -- see
        #: :meth:`_calibrate_charuco`.
        self.charuco_spec = charuco
        self.charuco_board = charuco.board() if charuco else None
        #: The board used to solve the *lens*, which has nothing to do with how
        #: this rig solves display space: intrinsics are a property of the
        #: camera. Keeping them separate is what lets the border route -- which
        #: asks the cluster for nothing -- still get its undistortion, from a
        #: printed board held in your hand that never goes near the screen.
        self.lens_spec = lens_board or charuco
        self.lens_board = self.lens_spec.board() if self.lens_spec else None
        self.charuco_anchor: np.ndarray | None = None
        #: Where the board sat in the frame the anchor was bound from, by corner
        #: id.  Compared against each later frame to catch the case the anchor
        #: cannot survive: the camera having moved since.
        self.charuco_bind_corners: dict[int, tuple[float, float]] = {}
        #: Views accumulated for an intrinsics solve, as correspondences rather
        #: than frames -- a phone photograph is tens of megabytes and its
        #: correspondences are a few kilobytes, and twenty of the former does
        #: not fit anywhere sensible.
        self._intrinsic_views: list[tuple[np.ndarray, np.ndarray]] = []
        self._intrinsic_frame_size: tuple[int, int] | None = None
        self.display_size = display_size or (
            calibration.geometry.display_size if calibration else (1920, 720)
        )
        #: Which way the display is being located, when the phone chooses. Each
        #: mode maps to its own display space: the corners of the screen for
        #: the corner modes, the board's canvas for the chessboard.
        self.calib_mode = ""
        self.corner_display_size = tuple(self.display_size)
        self.board_display_size = (tuple(board_display_size)
                                   if board_display_size else None)
        if calib_mode:
            self.set_mode(calib_mode)
        self.reference: np.ndarray | None = None
        self.reference_camera: np.ndarray | None = None
        self.history: list[CaptureRecord] = []

        if profile is not None and profile.reference_path:
            try:
                self.reference = profile.reference()
            except RuntimeError:
                self.reference = None

        if self.charuco_spec is not None:
            self._load_anchor()

    # -- calibration mode ---------------------------------------------------

    def available_modes(self) -> dict[str, bool]:
        """Which modes this session can offer, and why one might not be."""
        return {
            "manual": True,
            "auto": True,
            # The chessboard only means something with the board's own export:
            # it carries the exact corners the cluster drew. Guessing the board
            # is how a smaller grid inside a bigger one solves at the wrong scale.
            "chessboard": self.display_points is not None,
        }

    def set_mode(self, mode: str) -> None:
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"unknown mode {mode!r}; one of {', '.join(CALIBRATION_MODES)}")
        if not self.available_modes()[mode]:
            raise ValueError(
                "chessboard mode needs the board the cluster exports -- start "
                "with `layoutval go --board board.json`")
        with self.lock:
            switching = bool(self.calib_mode) and mode != self.calib_mode
            self.calib_mode = mode
            self.mark_corners = mode in ("manual", "auto")
            self.aperture = False
            self.display_size = (self.board_display_size or self.display_size
                                 if mode == "chessboard" else self.corner_display_size)
            if switching:
                # A new mode is a new calibration: the modes map into different
                # display spaces, so the old mapping and anything measured
                # through it no longer apply. Left in place, the page saw a
                # calibrated session with no reference and moved on to Reference
                # -- and the chessboard photo meant to calibrate became the
                # reference instead. The lens solve belongs to the phone, not to
                # the mode, so it stays.
                intrinsics = self.calibration.intrinsics if self.calibration else None
                self.calibration = None if intrinsics is None else Calibration(
                    intrinsics=intrinsics,
                    geometry=DisplayGeometry(H=np.eye(3), method=UNSOLVED,
                                             display_size=self.display_size),
                    drift_alarm_px=self.drift_alarm_px)
                self.reference = None
                self.reference_camera = None
                if self._profile_is_auto:
                    self.profile = None
                    self._profile_is_auto = False

    # -- helpers ------------------------------------------------------------

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.calibration and self.calibration.intrinsics:
            return Undistorter(self.calibration.intrinsics)(frame)
        return frame

    def status(self) -> dict[str, Any]:
        return {
            "calibrated": self.is_calibrated,
            "has_reference": self.reference is not None,
            "has_profile": self.profile is not None,
            "auto_inventory": self._profile_is_auto,
            "elements": len(self.profile) if self.profile else 0,
            "display_size": list(self.display_size),
            "fixed_camera": self.fixed_camera,
            "calibrates_from": self._calibrates_from(),
            "mode": self.calib_mode,
            "modes": self.available_modes() if self.calib_mode else {},
            "charuco": self.charuco_spec.to_dict() if self.charuco_spec else None,
            "anchor_bound": self.charuco_anchor is not None,
            "has_intrinsics": bool(
                self.calibration and self.calibration.intrinsics),
            "intrinsic_views": len(self._intrinsic_views),
            "intrinsic_views_wanted": INTRINSIC_VIEWS_WANTED,
            "uses_intrinsics": self._uses_intrinsics(),
            "needs_intrinsics": (
                self._uses_intrinsics()
                and not (self.calibration and self.calibration.intrinsics)),
            "intrinsics_rms": (
                round(self.calibration.intrinsics.rms, 3)
                if self.calibration and self.calibration.intrinsics else None),
            "sampling_ratio": (
                round(self.calibration.geometry.sampling_ratio(), 3)
                if self.calibration else None
            ),
            "captures": len(self.history),
        }

    @property
    def is_calibrated(self) -> bool:
        """Whether there is a real display-to-camera mapping, not a placeholder.

        Supplying intrinsics creates a Calibration so the undistortion has
        somewhere to live, and its identity homography rectifies to a raw crop
        of the camera frame. That crop is stable between reference and validate,
        so elements match beautifully and the numbers are in nothing at all --
        which is exactly the plausible-looking wrong answer this package exists
        to avoid. ``calibration is not None`` is not the question.
        """
        return (self.calibration is not None
                and self.calibration.geometry.method != UNSOLVED)

    def _uses_intrinsics(self) -> bool:
        # The chessboard gives the mapping from dozens of interior points and
        # never needed a lens solve; the routes that fit a display's edges do.
        if self.calib_mode == "chessboard":
            return False
        return self.charuco_spec is not None or self.aperture or self.mark_corners

    def _calibrates_from(self) -> str:
        if self.calib_mode:
            return {"manual": "corners you mark",
                    "auto": "corners found automatically",
                    "chessboard": "the cluster's chessboard"}[self.calib_mode]
        if self.mark_corners:
            return "corners you mark"
        if self.aperture:
            return "the display's own border"
        if self.charuco_spec is not None:
            return (
                "bezel markers" if self.charuco_anchor is not None
                else "bezel markers (needs binding first)"
            )
        return "screen content" if self.render is not None else "chessboard"

    def _store(self, name: str, img: np.ndarray) -> str:
        cv2.imwrite(str(self.out_dir / name), img)
        return name

    def _record(self, rec: CaptureRecord) -> CaptureRecord:
        self.history.insert(0, rec)
        del self.history[40:]
        return rec

    # -- actions ------------------------------------------------------------

    def handle(self, action: str, data: bytes,
               corners: list[list[float]] | None = None) -> CaptureRecord:
        frame = decode_upload(data)
        stamp = datetime.now().strftime("%H%M%S")
        base = f"{stamp}-{action}"
        if action == "propose":
            # Read-only: nothing about the session changes until the dots are
            # confirmed and come back as "corners".
            return self._propose(frame, base)
        with self.lock:
            if action == "corners":
                return self._record(self._calibrate_corners(frame, base, corners))
            if action == "intrinsics":
                return self._record(self._collect_intrinsics(frame, base))
            if action in ("calibrate", "rebind"):
                return self._record(self._calibrate(frame, base, rebind=action == "rebind"))
            if action == "reference":
                return self._record(self._reference(frame, base))
            if action == "validate":
                return self._record(self._validate(frame, base))
        raise ValueError(f"unknown action {action!r}")

    # -- the bezel board ----------------------------------------------------

    @property
    def _anchor_path(self) -> Path:
        return self.out_dir / "charuco-anchor.json"

    def _load_anchor(self) -> None:
        """Pick up an anchor bound in an earlier run, if it is for this board.

        A sticker on a bezel outlives the process that measured it, so the
        binding is worth keeping on disk.  It is only reusable for the *same*
        board though: the anchor is expressed in the board's own units, so
        loading one measured against a different grid or square size would
        silently rescale every measurement.  The spec is stored alongside it and
        checked rather than assumed.
        """
        if not self._anchor_path.exists():
            return
        try:
            saved = json.loads(self._anchor_path.read_text())
            spec = saved["spec"]
            anchor = np.asarray(saved["anchor"], dtype=np.float64).reshape(3, 3)
        except (OSError, ValueError, KeyError, TypeError):
            return
        assert self.charuco_spec is not None
        if spec != self.charuco_spec.to_dict():
            return
        if saved.get("display_size") != list(self.display_size):
            return
        self.charuco_anchor = anchor
        self.charuco_bind_corners = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in (saved.get("bind_corners") or {}).items()
        }

    def _save_anchor(self) -> None:
        assert self.charuco_spec is not None and self.charuco_anchor is not None
        self._anchor_path.write_text(json.dumps({
            "spec": self.charuco_spec.to_dict(),
            "anchor": self.charuco_anchor.tolist(),
            "display_size": list(self.display_size),
            "bind_corners": {
                str(k): [v[0], v[1]] for k, v in self.charuco_bind_corners.items()
            },
            "bound_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))

    def _board_pose_shift_px(self, undistorted: np.ndarray) -> float | None:
        """How far the board has moved in the frame since the anchor was bound.

        The anchor is a fixed board-to-display transform, and it is exactly that
        only while the view it was measured from still holds.  Reused from a
        different pose it stays *plausible* and gets worse, which is the failure
        mode this package exists to avoid.  Comparing the board's own corners
        between the two frames costs nothing and turns it into something the
        record can say out loud.
        """
        if not self.charuco_bind_corners:
            return None
        try:
            corners, ids = detect_charuco(undistorted, self.charuco_board)
        except RuntimeError:
            return None
        shifts = [
            float(np.hypot(*(corners[i] - self.charuco_bind_corners[int(cid)])))
            for i, cid in enumerate(ids)
            if int(cid) in self.charuco_bind_corners
        ]
        return float(np.median(shifts)) if shifts else None

    def _geometry_for_bind(self, undistorted: np.ndarray) -> Any:
        """Display-to-camera for the binding frame, by whatever route there is.

        Binding needs the active area located *independently* of the markers --
        that is the whole content of the measurement.  Anything that does it
        will do, so this takes the same routes ``calibrate`` takes without the
        board: the framebuffer when it was supplied, the cluster's chessboard
        otherwise.
        """
        if self.render is not None:
            return homography_from_screen_content(
                self.render, undistorted, display_size=self.display_size
            )
        points = (
            self.display_points if self.display_points is not None
            else chessboard_display_points(
                self.pattern_size, self.square_px, self.pattern_origin
            )
        )
        return homography_from_display_pattern(
            undistorted, self.pattern_size, points, display_size=self.display_size
        )

    def _calibrate_charuco(
        self, frame: np.ndarray, undistorted: np.ndarray, rec: CaptureRecord,
        *, rebind: bool,
    ) -> CaptureRecord:
        """Calibrate from a board fixed to the bezel.

        Two different operations wear one button here, and which one runs
        depends on whether the board has been tied to the active area yet.

        **Binding**, the first time: the frame has to show the markers *and*
        something that locates the active area on its own -- the cluster's
        calibration screen, or its framebuffer via ``--render``.  From those two
        together comes the board-to-display transform, and that is the only time
        the cluster has to co-operate.

        **Using it**, every time after: the markers alone are enough, and the
        screen can show whatever is under test.  This is the point of the whole
        exercise, because a real cluster will not draw a chessboard on request.
        """
        assert self.charuco_spec is not None
        # Undistortion is not optional on this route, and that is measured
        # rather than assumed.  A homography cannot represent lens distortion,
        # and unlike a board on the screen the markers sit in a different part
        # of the frame from the active area -- so the board fit and the display
        # fit are each locally wrong in a different direction and the errors
        # compound instead of cancelling.  Measured on the simulated bezel
        # (benchmarks/bezel_anchor.py), median over five poses: undistorted the
        # route holds 0.10 px whatever the lens, and with the distortion left in
        # it runs 0.6 px on a mild lens to 5.7 px on a very wide one -- while
        # elements start failing to match at all, up to every one of them.  So
        # refuse: the failure is lens-dependent, so a frame that looks fine here
        # says nothing about the next camera.
        if not (self.calibration and self.calibration.intrinsics):
            rec.verdict = "FAILED"
            rec.detail = (
                "the bezel-marker route needs camera intrinsics and none were "
                "given. The markers are photographed away from the screen's own "
                "part of the frame, so lens distortion does not cancel the way "
                "it nearly does for a board on the screen: measured, this route "
                "holds 0.10 px undistorted on any lens, and 0.6 px to 5.7 px "
                "with the distortion left in -- losing elements from the match "
                "entirely on the wider ones. Run "
                "`layoutval calibrate-intrinsics shots/*.jpg` once for this "
                "camera and lens, then restart with --intrinsics."
            )
            return rec

        if rebind:
            self.charuco_anchor = None
            self.charuco_bind_corners = {}

        if self.charuco_anchor is None:
            return self._bind_charuco(frame, undistorted, rec)
        return self._use_charuco(undistorted, rec)

    def _bind_charuco(
        self, frame: np.ndarray, undistorted: np.ndarray, rec: CaptureRecord
    ) -> CaptureRecord:
        try:
            geometry = self._geometry_for_bind(undistorted)
        except RuntimeError:
            rec.verdict = "FAILED"
            rec.detail = (
                "the bezel board has not been tied to the active area yet, and "
                "this frame cannot do it: " + self._why_no_board(frame, rec.name)
                + " Binding is the one shot that needs both in view at once -- "
                "the markers and the screen showing its calibration pattern. "
                "After it, the markers alone are enough and the screen is free."
            )
            return rec
        try:
            anchor = charuco_anchor(undistorted, self.charuco_board, geometry)
            corners, ids = detect_charuco(undistorted, self.charuco_board)
        except RuntimeError as exc:
            rec.verdict = "FAILED"
            rec.detail = (
                f"the screen was located but the bezel board was not: {exc}. "
                f"The frame is saved as {rec.name} -- both have to be in this "
                "one frame, in focus, for binding to mean anything."
            )
            return rec

        self.charuco_anchor = anchor
        self.charuco_bind_corners = {
            int(cid): (float(corners[i][0]), float(corners[i][1]))
            for i, cid in enumerate(ids)
        }
        self._save_anchor()
        self._commit_geometry(geometry, "phone capture, binding the bezel board")
        rec.verdict = "OK"
        rec.detail = (
            f"bound the bezel board to the active area from {len(ids)} board "
            f"corners; the screen itself solved to {geometry.residual_px:.3f} "
            f"camera px. From here on Calibrate needs only the markers, so the "
            f"cluster can stay on the screen under test. Saved as "
            f"{self._anchor_path.name}, so later runs pick it up. Re-bind if the "
            "sticker is ever moved or reprinted"
        )
        self._note_sampling_ratio(rec, geometry)
        return rec

    def _use_charuco(self, undistorted: np.ndarray, rec: CaptureRecord) -> CaptureRecord:
        try:
            geometry = homography_from_charuco(
                undistorted, self.charuco_board, self.charuco_anchor,
                display_size=self.display_size,
            )
        except RuntimeError as exc:
            rec.verdict = "FAILED"
            rec.detail = (
                f"{exc}. The frame is saved as {rec.name}. The whole board has "
                "to be in view and readable -- this route never needs the screen "
                "to show anything, but it does need the markers."
            )
            return rec
        self._commit_geometry(geometry, "phone capture, bezel board")
        rec.verdict = "OK"
        rec.detail = (
            f"solved from the bezel markers alone -- nothing was asked of the "
            f"screen. Board fit residual {geometry.residual_px:.3f} camera px"
        )
        shift = self._board_pose_shift_px(undistorted)
        if shift is not None:
            rec.detail += f", board {shift:.0f} camera px from where it was bound"
            if shift > ANCHOR_POSE_TOLERANCE_PX:
                # Measured, on the simulated bezel: re-bound in the pose it is
                # used in, this route matches a board on the screen (0.09 px
                # median against 0.05 px).  Carried across a camera move it
                # runs 0.25-0.32 px median and 0.56-0.94 px at p95.  It stays
                # usable; it stops being sub-tenth-pixel, and a tolerance set
                # from the first figure does not hold for the second.
                rec.detail += (
                    ". That is far enough that the anchor is being used from a "
                    "different view than it was measured in, which costs "
                    "accuracy: about 0.3 px typical and 0.9 px at the tail, "
                    "against under 0.1 px when it is re-bound in the pose it is "
                    "used in. Re-bind from here, or widen the tolerance to suit"
                )
        self._note_sampling_ratio(rec, geometry)
        return rec

    def _commit_geometry(self, geometry: Any, source: str) -> None:
        """Adopt a new display-to-camera solve and drop what it invalidates."""
        # Display space is whatever this solve mapped into; with modes that map
        # into different spaces, the session follows the one in force.
        self.display_size = tuple(geometry.display_size)
        self.calibration = Calibration(
            intrinsics=self.calibration.intrinsics if self.calibration else None,
            geometry=geometry,
            rig={"source": source},
            drift_alarm_px=self.drift_alarm_px,
        )
        self.calibration.save(self.out_dir / "calibration.json")
        # The geometry has moved, so anything measured against the old one is
        # meaningless.  Drop it rather than let it be compared across.
        self.reference = None
        self.reference_camera = None

    def _note_sampling_ratio(self, rec: CaptureRecord, geometry: Any) -> None:
        ratio = geometry.sampling_ratio()
        rec.detail += f", sampling ratio {ratio:.2f} camera px per display px"
        if ratio < 2.0:
            rec.detail += (
                ". Below 2 you cannot reliably resolve a one-display-pixel "
                "shift -- move closer or zoom in before trusting a tolerance"
            )

    def _collect_intrinsics(self, frame: np.ndarray, base: str) -> CaptureRecord:
        """Accumulate one view towards a lens solve, from the phone.

        Intrinsics are per camera and lens, and getting them has meant shooting
        a set of board photographs and moving the files onto the laptop by hand
        -- which is the step most likely to stop somebody before they start, and
        an odd one to insist on when a phone is already uploading frames to this
        process over the network.

        The board can be a printed one or the cluster's own calibration screen:
        what a lens solve needs is a *planar* target of known geometry seen from
        many angles, and a flat panel showing a chessboard is exactly that.
        """
        rec = CaptureRecord(name=f"{base}.jpg", action="intrinsics",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        h, w = frame.shape[:2]

        if self._intrinsic_frame_size is None:
            self._intrinsic_frame_size = (w, h)
            if self.calibration and self.calibration.intrinsics:
                # Shooting a fresh set is how you ask for a fresh solve. The
                # loaded one stays in use until the new set actually solves, so
                # a half-finished re-shoot never leaves the rig with no lens.
                rec.detail = "re-solving the lens. "
        elif self._intrinsic_frame_size != (w, h):
            # Intrinsics are in pixels, so they belong to one frame size. Mixing
            # sizes silently produces a calibration that fits neither.
            rec.verdict = "FAILED"
            rec.detail = (
                f"this frame is {w}x{h} but the others are "
                f"{self._intrinsic_frame_size[0]}x{self._intrinsic_frame_size[1]}. "
                "Intrinsics are measured in pixels, so every view has to come "
                "from the same camera at the same resolution -- check the phone "
                "is not switching lens between shots, and start again."
            )
            return rec

        try:
            obj, img = board_points_for_intrinsics(
                frame,
                board=self.lens_board,
                pattern_size=self.pattern_size,
            )
        except (RuntimeError, ValueError) as exc:
            rec.verdict = "FAILED"
            if self.lens_spec is not None:
                wanted = (f"the {self.lens_spec.squares_x}x"
                          f"{self.lens_spec.squares_y} ChArUco board")
            else:
                wanted = (f"a {self.pattern_size[0]}x{self.pattern_size[1]} "
                          "chessboard")
            rec.detail = (
                f"{exc}. Not counted. This step is looking for {wanted} -- "
                "not the cluster. It is the camera's lens being measured here, "
                "not the rig, so the target is a board you hold in front of the "
                "phone and it never goes near the screen under test. No "
                "printer? Open the board image full-screen on any monitor and "
                "photograph that; a screen is as flat a target as paper. Fill "
                f"the frame with it and keep it sharp. Saved as {rec.name}."
            )
            return rec

        self._intrinsic_views.append((obj, img))
        got, want = len(self._intrinsic_views), INTRINSIC_VIEWS_WANTED
        rec.verdict = "OK"
        rec.detail = (rec.detail or "") + (
            f"view {got} of {want}. Change the angle between shots.")
        if got < want:
            return rec

        try:
            intrinsics = solve_intrinsics(
                [o for o, _ in self._intrinsic_views],
                [i for _, i in self._intrinsic_views],
                self._intrinsic_frame_size,
                min_views=want,
            )
        except (RuntimeError, cv2.error) as exc:
            rec.verdict = "FAILED"
            rec.detail = f"the solve did not converge: {exc}. Shoot more views."
            return rec

        # Scored on whether it generalises, not on how well it fits. A solve
        # can report 0.05 px reprojection error and leave 1.3 px of real error
        # behind, because reprojection only measures the fit to the views it
        # was handed -- and a set of near-identical views is fitted beautifully
        # and is wrong everywhere else. See LensCheck.
        check = check_intrinsics(
            [o for o, _ in self._intrinsic_views],
            [i for _, i in self._intrinsic_views],
            self._intrinsic_frame_size,
        )
        complaint = check.complaint()
        implied = check.expected_element_error_px()
        rec.detail = (
            f"lens solved from {got} views: {check.quality()}, "
            f"about {implied:.2f} px accuracy"
        )
        if complaint is None:
            self.calibration = Calibration(
                intrinsics=intrinsics,
                geometry=(self.calibration.geometry if self.calibration
                          else DisplayGeometry(H=np.eye(3), method=UNSOLVED,
                                               display_size=self.display_size)),
                rig={"source": "phone capture"},
                drift_alarm_px=self.drift_alarm_px,
            )
            out = self.out_dir / "intrinsics.json"
            out.write_text(json.dumps(intrinsics.to_dict(), indent=2))
            rec.detail += ". Ready -- go to Calibrate."
            if implied > 0.40:
                # Adopted, because even a loose model beats none: skipping
                # undistortion on the same lens costs 4.08 px where the loosest
                # solve measured cost 1.30. Worth one clause, not a paragraph.
                rec.detail += " A printed board would tighten it."
            self._intrinsic_views.clear()
            self._intrinsic_frame_size = None
            return rec

        # Not adopted. A lens model this loose is worse than none -- measured,
        # one that passed the old rms gate left more error behind than skipping
        # undistortion altogether -- so the views are kept and the set can be
        # extended rather than silently accepted.
        rec.verdict = "FAILED"
        rec.detail = f"{got} views: {complaint.split('.')[0]}. Keep shooting."
        return rec

    def _calibrate(
        self, frame: np.ndarray, base: str, *, rebind: bool = False
    ) -> CaptureRecord:
        rec = CaptureRecord(name=f"{base}.jpg",
                            action="rebind" if rebind else "calibrate",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        undistorted = self._undistort(frame)
        if self.aperture and not rebind:
            return self._calibrate_from_aperture(undistorted, rec)
        if self.charuco_spec is not None:
            return self._calibrate_charuco(frame, undistorted, rec, rebind=rebind)
        if rebind:
            rec.verdict = "FAILED"
            rec.detail = (
                "there is no bezel board to re-bind. Re-binding ties a board "
                "fixed to the bezel back to the active area; start the server "
                "with --charuco to use one."
            )
            return rec
        if self.render is not None:
            return self._calibrate_from_content(undistorted, rec)
        # Prefer the board's own exported corners over reconstructing them.
        # The HMI writes exactly where it drew them, which sidesteps both
        # guessing the square size and the half-pixel between Qt's convention
        # and the detector's -- and that half pixel goes straight into the
        # homography and from there into every measurement through it.
        points = (
            self.display_points if self.display_points is not None
            else chessboard_display_points(
                self.pattern_size, self.square_px, self.pattern_origin
            )
        )
        try:
            geometry = homography_from_display_pattern(
                undistorted, self.pattern_size, points, display_size=self.display_size
            )
        except RuntimeError:
            rec.verdict = "FAILED"
            rec.detail = self._why_no_board(frame, rec.name)
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
        # A chessboard detector will find a smaller grid inside a bigger one, so
        # a pattern size that does not match what was drawn need not fail: it can
        # solve cleanly at the wrong scale, and nothing downstream notices. There
        # is no test for it here -- describing a 3x3 board of 40 px squares is a
        # self-consistent reading of a 9x6 one, and this end cannot know which
        # was drawn. Only the board's own export can settle it, so say plainly
        # when it was not used rather than imply the question was checked.
        if self.display_points is None:
            rec.detail += (
                ". Note: the board was described by hand rather than taken from "
                "--board, so nothing here can confirm it is the board the cluster "
                "actually drew -- a detector will match a smaller grid inside a "
                "larger one and solve cleanly at the wrong scale. Prefer --board "
                "with the file the cluster exported"
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

    def _propose(self, frame: np.ndarray, base: str) -> CaptureRecord:
        """Proposed display corners, for the phone to show as draggable dots."""
        rec = CaptureRecord(name=f"{base}.jpg", action="propose",
                            when=datetime.now().isoformat(timespec="seconds"))
        intrinsics = self.calibration.intrinsics if self.calibration else None
        if intrinsics is None:
            proposal = propose_display_corners(frame)
        else:
            # Found on the undistorted frame, where the display's edges are
            # straight. On the raw photograph lens distortion bows them, and
            # straight lines fitted to bowed edges met 29-34 px off at the
            # corners -- enough that the snap-to-edge step then refused them.
            # The dots go back to the phone in the raw photograph's pixels,
            # since that is the picture it is showing.
            undistorter = Undistorter(intrinsics)
            proposal = propose_display_corners(undistorter(frame))
            if proposal is not None:
                proposal.corners = _distort_points(
                    proposal.corners, intrinsics, undistorter.new_K)
        if proposal is None:
            rec.verdict = "FAILED"
            rec.detail = "couldn't find the display -- tap its four corners."
            return rec
        rec.verdict = "OK"
        rec.detail = ("found the display. Check the dots, drag any that are off, "
                      "then use them." if not proposal.confident else
                      "found the display. Use these corners, or drag to adjust.")
        rec.proposal = proposal.to_dict(frame.shape)
        return rec

    def _calibrate_corners(
        self, frame: np.ndarray, base: str,
        corners: list[list[float]] | None,
    ) -> CaptureRecord:
        """Calibrate from four corners somebody pointed at.

        The one route that always works, because the hard part -- deciding
        which rectangle in the picture is the display -- is done by a person,
        instantly, and the part people are bad at is done here: the taps are
        only a guide, and the edges are fitted to the real intensity step.
        """
        rec = CaptureRecord(name=f"{base}.jpg", action="corners",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        if not corners or len(corners) != 4:
            rec.verdict = "FAILED"
            rec.detail = "tap all four corners of the display, then shoot."
            return rec

        undistorted = self._undistort(frame)
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if self.calibration and self.calibration.intrinsics:
            # The taps are on the frame as photographed; the measurement runs
            # on the undistorted one, so the points have to make the same trip.
            undistorter = Undistorter(self.calibration.intrinsics)
            points = cv2.undistortPoints(
                points.reshape(-1, 1, 2), self.calibration.intrinsics.K,
                self.calibration.intrinsics.dist, P=undistorter.new_K,
            ).reshape(-1, 2)

        try:
            geometry = homography_from_marked_corners(
                undistorted, points, display_size=self.corner_display_size,
                inset_px=self.display_inset_px,
            )
        except RuntimeError as exc:
            rec.verdict = "FAILED"
            rec.detail = f"{str(exc).split('.')[0]}."
            return rec

        self._store(rec.name.replace(".jpg", "-corners.jpg"),
                    draw_aperture(undistorted, geometry.corners))
        self._commit_geometry(geometry, "phone capture, corners marked by hand")
        rec.verdict = "OK"
        rec.detail = "display located from your four corners. Next: Reference."
        if geometry.method.endswith("(unrefined)"):
            rec.detail += (
                " The edges could not be snapped to the panel boundary, so this "
                "is only as good as the taps were -- check "
                + rec.name.replace(".jpg", "-corners.jpg")
            )
        ratio = geometry.sampling_ratio()
        if ratio < 2.0:
            rec.detail += f" (sampling {ratio:.1f} -- move closer for finer work)"
        return rec

    def _calibrate_from_aperture(
        self, undistorted: np.ndarray, rec: CaptureRecord
    ) -> CaptureRecord:
        """Calibrate from the display's own border, asking the cluster nothing.

        Every other route wants something: a pattern drawn, a white frame, the
        framebuffer, or -- for the bezel markers -- one binding frame with the
        calibration screen up.  This one wants nothing at all, because the
        aperture is hardware and is in every photograph whatever the software is
        doing.
        """
        if not (self.calibration and self.calibration.intrinsics):
            rec.verdict = "FAILED"
            rec.detail = ("the lens is not solved yet. Do Intrinsics first.")
            return rec
        try:
            geometry = homography_from_display_aperture(
                undistorted, display_size=self.display_size,
                inset_px=self.display_inset_px,
            )
        except RuntimeError as exc:
            rec.verdict = "FAILED"
            rec.detail = f"{str(exc).split('.')[0]}. See {rec.name}."
            return rec
        # Always leave a picture of what was found. This route fails by picking
        # a plausible wrong rectangle, and no number describes that as well as
        # looking at it does.
        try:
            corners, _ = find_display_aperture(
                undistorted, display_size=self.display_size)
            self._store(rec.name.replace(".jpg", "-aperture.jpg"),
                        draw_aperture(undistorted, corners))
        except RuntimeError:
            pass
        self._commit_geometry(geometry, "phone capture, the display's own border")
        diag = geometry.aperture
        rec.verdict = "OK"
        rec.detail = "found the display's border. Next: Reference."
        ratio = geometry.sampling_ratio()
        if ratio < 2.0:
            rec.detail += f" (sampling {ratio:.1f} -- move closer for finer work)"
        return rec

    def _calibrate_from_content(
        self, undistorted: np.ndarray, rec: CaptureRecord
    ) -> CaptureRecord:
        """Calibrate against the framebuffer, with no pattern on the screen."""
        try:
            geometry = homography_from_screen_content(
                self.render, undistorted, display_size=self.display_size
            )
        except RuntimeError as exc:
            rec.verdict = "FAILED"
            rec.detail = (
                f"{exc}. The frame is saved as {rec.name} -- open it and see what "
                "it caught."
            )
            return rec

        self.calibration = Calibration(
            intrinsics=self.calibration.intrinsics if self.calibration else None,
            geometry=geometry,
            rig={"source": "phone capture, matched to the framebuffer"},
            drift_alarm_px=self.drift_alarm_px,
        )
        self.calibration.save(self.out_dir / "calibration.json")
        ratio = geometry.sampling_ratio()
        rec.verdict = "OK"
        rec.detail = (
            f"matched the screen's own content -- no pattern needed. "
            f"{getattr(geometry, 'inliers', 0)} of "
            f"{getattr(geometry, 'matches', 0)} matches agreed, sampling ratio "
            f"{ratio:.2f} camera px per display px"
        )
        if geometry.method == "screen_content(unrefined)":
            rec.detail += (
                ". The refinement step did not converge, so this is the feature "
                "fit alone and is the looser of the two -- re-shoot squarer on, "
                "or in better focus"
            )
        if ratio < 2.0:
            rec.detail += (
                ". Below 2 camera px per display px you cannot reliably resolve a "
                "one-display-pixel shift -- move closer or zoom in before trusting "
                "a tolerance"
            )
        self.reference = None
        self.reference_camera = None
        return rec

    def _why_no_board(self, frame: np.ndarray, saved_as: str) -> str:
        """Why a calibration frame had no chessboard in it.

        Overwhelmingly the answer is that the cluster was not showing one --
        selecting Calibrate on the phone says how to read the photograph, it
        does not put a board on the screen. That is worth distinguishing from a
        board that is present but unreadable, because the two have nothing to do
        with each other, and the frame itself is on disk either way.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, spread = float(gray.mean()), float(gray.std())
        wanted = f"{self.pattern_size[0]}x{self.pattern_size[1]} inner corners"
        where = f"The frame is saved as {saved_as} -- open it and see what it caught."

        # A chessboard filling the frame is half black and half white, so it has
        # a wide spread. A cluster screen is mostly dark background.
        if spread < 45:
            return (
                "no chessboard here, and this frame does not look like one: it is "
                f"{'very dark' if mean < 50 else 'low in contrast'} "
                f"(mean {mean:.0f}, spread {spread:.0f} of 255). "
                "Is the cluster actually showing the pattern? Selecting Calibrate "
                "here only says how to read the photograph; the cluster has to be "
                f"put on its chessboard separately. {where}"
            )
        return (
            f"a board may be there but it did not read as {wanted}. Fill the frame "
            "with it, square on, with the whole board and a margin around it "
            "visible, and watch for glare on the glass. If the cluster is drawing "
            "a different board, re-export it and restart with that --board file. "
            f"{where}"
        )

    def _reference(self, frame: np.ndarray, base: str) -> CaptureRecord:
        rec = CaptureRecord(name=f"{base}.jpg", action="reference",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        if not self.is_calibrated:
            rec.verdict = "FAILED"
            rec.detail = "do Calibrate first -- there is no display mapping yet."
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

        # With no authored inventory, take one from the frame itself rather than
        # having nothing to measure. Only ever replaces an inventory this made
        # earlier -- an authored profile is the better answer and is left alone.
        if self.auto_profile and (self.profile is None or self._profile_is_auto):
            found = profile_from_reference(
                self.reference,
                screen=self.profile.screen if self.profile else "auto",
                display_size=self.display_size,
            )
            self.profile = found
            self._profile_is_auto = True
            rec.detail += (
                f". No layout profile was loaded, so {len(found)} element(s) were "
                "found in this frame and will be measured against it. That answers "
                "whether a later frame matches this one, not whether the build "
                "matches the design, and it treats everything as fixed -- keep the "
                "cluster in the state it is in now"
            )
        return rec

    def _validate(self, frame: np.ndarray, base: str) -> CaptureRecord:
        rec = CaptureRecord(name=f"{base}.jpg", action="validate",
                            when=datetime.now().isoformat(timespec="seconds"))
        self._store(rec.name, frame)
        if not self.is_calibrated:
            rec.verdict = "FAILED"
            rec.detail = "do Calibrate first -- there is no display mapping yet."
            return rec
        if self.reference is None:
            rec.verdict = "FAILED"
            rec.detail = "take a reference first: there is nothing to measure against"
            return rec
        if self.profile is None:
            rec.verdict = "FAILED"
            rec.detail = (
                "no inventory to measure. Take a reference first -- with no "
                "--profile, the elements are found in the reference frame"
            )
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
                # A note, not a finding: this is how a hand-held rig always
                # works, so it is true of every frame and says nothing about
                # this one. Marking it "review" made every hand-held run come
                # back REVIEW no matter how cleanly it measured.
                report.flag(
                    "pose_resolved", severity="note",
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
.modes{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:14px}
.modes button{background:#1E272C;color:#C2CED6;border:1px solid #2B373D;border-radius:9px;
              padding:10px 4px;font-size:13px;font-weight:600;-webkit-tap-highlight-color:transparent}
.modes button.on{background:#1D5B8A;border-color:#3E8FD0;color:#fff}
.modes button:disabled{color:#4A5961;border-style:dashed}
#markcv{touch-action:none}
button.ghost{flex:1;background:#1E272C;color:#E3EAE8;border:1px solid #2B373D;
             border-radius:9px;padding:13px;font-size:15px;font-weight:600}
button.ghost:disabled{color:#5C6B73}
button.ghost#marksend:not(:disabled){background:#2F7D57;border-color:#2F7D57}
</style>
</head>
<body>
<h1>Cluster capture</h1>
<p class="sub">Point at the cluster and shoot. Measuring happens on the laptop.</p>

<div class="modes" id="modebar" style="display:none">
  <button data-m="auto">Auto corners</button>
  <button data-m="manual">Tap corners</button>
  <button data-m="chessboard">Chessboard</button>
</div>
<div class="hint" id="modehint" style="margin:-6px 0 12px"></div>

<div class="steps" id="steps">
  <div class="step" data-a="corners"><b>1 Corners</b><span>tap the display</span></div>
  <div class="step" data-a="calibrate" style="display:none"><b>1 Calibrate</b><span>chessboard up</span></div>
  <div class="step" data-a="reference"><b>2 Reference</b><span>correct screen</span></div>
  <div class="step" data-a="validate"><b>3 Validate</b><span>screen under test</span></div>
</div>

<div class="steps" id="introw" style="display:none;grid-template-columns:1fr">
  <div class="step" data-a="intrinsics"><b>0 Intrinsics</b>
    <span id="intcount">the lens, once per phone</span></div>
</div>

<div class="steps" id="rebindRow" style="display:none;grid-template-columns:1fr">
  <div class="step" data-a="rebind"><b>Re-bind the bezel board</b>
    <span>only after the sticker moves</span></div>
</div>

<div class="card">
  <label class="shoot" id="shootLabel" for="shot">Take photo</label>
  <input id="shot" type="file" accept="image/*" capture="environment">
  <div class="hint" id="hint"></div>
</div>

<div class="card" id="marker" style="display:none">
  <b id="marktitle">Tap the four corners of the display</b>
  <div class="hint" id="markhint">corner 1 of 4</div>
  <div style="position:relative;margin-top:10px">
    <img id="markimg" class="shot" style="margin:0">
    <canvas id="markcv" style="position:absolute;left:0;top:0;width:100%;height:100%"></canvas>
  </div>
  <div style="display:flex;gap:8px;margin-top:10px">
    <button id="markundo" class="ghost">Undo</button>
    <button id="marksend" class="ghost" disabled>Use these corners</button>
  </div>
</div>

<div class="card" id="result" style="display:none"></div>

<div class="card">
  <div class="stat"><span>Calibrated</span><b id="s-cal">—</b></div>
  <div class="stat"><span>Reference</span><b id="s-ref">—</b></div>
  <div class="stat"><span>Elements</span><b id="s-el">—</b></div>
  <div class="stat"><span>Solves from</span><b id="s-from">—</b></div>
  <div class="stat"><span>Sampling ratio</span><b id="s-sr">—</b></div>
  <div class="stat"><span>Captures</span><b id="s-n">—</b></div>
</div>

<script>
const TOKEN = new URLSearchParams(location.search).get("t") || "";
let action = "calibrate";
let BEZEL = false, ANCHORED = false;
let MODE = "", MODES = {};
const $ = id => document.getElementById(id);

document.querySelectorAll(".step").forEach(el => {
  el.onclick = () => { action = el.dataset.a; paintSteps(); };
});
function paintSteps() {
  document.querySelectorAll(".step").forEach(el =>
    el.classList.toggle("on", el.dataset.a === action));
  const hints = {
    calibrate: "Put the cluster on its chessboard pattern first. Fill the frame with it, square on.",
    reference: "Show the screen under test, correct. This becomes what later shots are compared against.",
    validate: "Show the same screen with whatever you are testing. Keep the phone where it was.",
    rebind: "Only needed if the bezel sticker was moved or reprinted. Needs the markers and the calibration screen together again.",
    corners: MODE === "auto"
      ? "Shoot the whole screen with a margin round it. The corners are found for you \u2014 check them, drag any that are off. Once per camera position."
      : "Shoot the cluster, then tap its four corners on the photo. Rough taps are fine \u2014 the edges get snapped to the panel. Once per camera position.",
    intrinsics: "Shoot the board from a different angle and distance each time \u2014 near, far, tilted, and in each corner of the frame. Shots that all look alike cannot separate the lens from the pose."
  };
  if (BEZEL) {
    hints.calibrate = ANCHORED
      ? "Get the whole bezel board in frame. The screen can show anything \u2014 the markers do the work."
      : "First time only: the markers and the cluster's calibration screen, together in one frame.";
  }
  $("hint").textContent = hints[action];
}

document.querySelectorAll(".modes button").forEach(b => {
  b.onclick = async () => {
    if (b.disabled || b.dataset.m === MODE) return;
    try {
      const r = await fetch(`/mode?t=${TOKEN}`, {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({mode: b.dataset.m})});
      const s = await r.json();
      if (!r.ok) { show({verdict: "FAILED", detail: s.detail}); return; }
    } catch (e) { show({verdict: "FAILED", detail: "could not reach the laptop: " + e}); }
    $("marker").style.display = "none";
    action = b.dataset.m === "chessboard" ? "calibrate" : "corners";
    await refresh(); paintSteps();
  };
});

function paintModes() {
  const bar = $("modebar");
  bar.style.display = MODE ? "grid" : "none";
  document.querySelectorAll(".modes button").forEach(b => {
    b.classList.toggle("on", b.dataset.m === MODE);
    b.disabled = MODES[b.dataset.m] === false;
  });
  $("modehint").textContent = !MODE ? "" : {
    auto: "The corners are found for you. Check the dots, drag any that are off.",
    manual: "You tap the four corners of the screen on the photo.",
    chessboard: "The cluster shows its chessboard for one shot."
  }[MODE] + (MODES.chessboard === false && MODE !== "chessboard"
      ? " (Chessboard needs --board.)" : "");
}

async function refresh() {
  try {
    const r = await fetch(`/status?t=${TOKEN}`);
    const s = await r.json();
    $("s-cal").textContent = s.calibrated ? "yes" : "no";
    $("s-ref").textContent = s.has_reference ? "yes" : "no";
    $("s-el").textContent = s.has_profile
      ? s.elements + (s.auto_inventory ? " (from reference)" : "")
      : "after reference";
    $("s-sr").textContent = s.sampling_ratio ? s.sampling_ratio.toFixed(2) : "—";
    $("s-n").textContent = s.captures;
    $("s-from").textContent = s.calibrates_from;
    const was = BEZEL + "/" + ANCHORED;
    BEZEL = !!s.charuco; ANCHORED = !!s.anchor_bound;
    $("rebindRow").style.display = (BEZEL && ANCHORED) ? "grid" : "none";
    $("introw").style.display = s.uses_intrinsics ? "grid" : "none";
    MODE = s.mode || ""; MODES = s.modes || {};
    paintModes();
    document.querySelector('[data-a="corners"] span').textContent =
      MODE === "auto" ? "find the display" : "tap the display";
    const byHand = MODE ? MODE !== "chessboard"
                        : s.calibrates_from === "corners you mark";
    document.querySelector('[data-a="corners"]').style.display = byHand ? "" : "none";
    document.querySelector('[data-a="calibrate"]').style.display = byHand ? "none" : "";
    document.querySelector('[data-a="corners"]').classList.toggle("done", s.calibrated);
    if (byHand && action === "calibrate") { action = "corners"; paintSteps(); }
    if (!byHand && action === "corners") { action = "calibrate"; paintSteps(); }
    $("intcount").textContent = s.intrinsic_views
      ? `${s.intrinsic_views} of ${s.intrinsic_views_wanted} views`
      : (s.has_intrinsics ? "solved \u2014 tap to redo" : "the lens, once per phone");
    document.querySelector('[data-a="intrinsics"]').classList
      .toggle("done", !!s.has_intrinsics);
    // Nothing else can run until the lens is solved, so start there rather
    // than letting Calibrate be picked and refused.
    if (s.needs_intrinsics && (action === "calibrate" || action === "corners")) {
      action = "intrinsics";
    }
    if (!s.needs_intrinsics && action === "intrinsics" && !s.has_intrinsics) {
      action = byHand ? "corners" : "calibrate";
    }
    document.querySelector('[data-a="calibrate"] span').textContent =
      BEZEL ? (ANCHORED ? "markers only" : "bind: markers + pattern") : "chessboard up";
    if (was !== BEZEL + "/" + ANCHORED) paintSteps();
    document.querySelector('[data-a="calibrate"]').classList.toggle("done", s.calibrated);
    document.querySelector('[data-a="reference"]').classList.toggle("done", s.has_reference);
    if (s.calibrated && !s.has_reference && (action === "calibrate" || action === "corners")) {
      action = "reference"; paintSteps();
    }
  } catch (e) { /* the laptop went away; the next poll will say so */ }
}

// --- marking the display's corners by hand -------------------------------
let taps = [], pending = null;

function paintTaps() {
  const cv = $("markcv"), img = $("markimg");
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  g.strokeStyle = "#4FB584"; g.fillStyle = "#4FB584"; g.lineWidth = 2;
  taps.forEach((t, i) => {
    const x = t[0] * cv.width, y = t[1] * cv.height;
    g.beginPath(); g.arc(x, y, 9, 0, 7); g.fill();
    g.fillStyle = "#0F1518"; g.font = "bold 12px system-ui";
    g.fillText(String(i + 1), x - 3, y + 4); g.fillStyle = "#4FB584";
  });
  if (taps.length > 1) {
    g.beginPath();
    taps.forEach((t, i) => {
      const x = t[0] * cv.width, y = t[1] * cv.height;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    if (taps.length === 4) g.closePath();
    g.stroke();
  }
  $("markhint").textContent = taps.length < 4
    ? `corner ${taps.length + 1} of 4 \u2014 go round the display, any direction`
    : "Press and drag a dot to move it. Close is fine \u2014 the edges get snapped to the panel.";
  $("marksend").disabled = taps.length !== 4;
}

// Tap to place a dot, press on a dot to drag it. Same for proposed dots, so a
// proposal that is slightly off is fixed with a thumb rather than retaken.
let dragging = -1;
function at(ev) {
  const r = $("markcv").getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height, r];
}
$("markcv").addEventListener("pointerdown", ev => {
  const [x, y, r] = at(ev);
  let best = -1, bestD = 1e9;
  taps.forEach((t, i) => {
    const d = Math.hypot((t[0] - x) * r.width, (t[1] - y) * r.height);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best >= 0 && bestD < 34) {
    dragging = best;
    $("markcv").setPointerCapture(ev.pointerId);
  } else if (taps.length < 4) {
    taps.push([x, y]);
    paintTaps();
  }
  ev.preventDefault();
});
$("markcv").addEventListener("pointermove", ev => {
  if (dragging < 0) return;
  const [x, y] = at(ev);
  taps[dragging] = [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  paintTaps();
  ev.preventDefault();
});
["pointerup", "pointercancel"].forEach(k =>
  $("markcv").addEventListener(k, () => { dragging = -1; }));
$("markundo").onclick = () => { taps.pop(); paintTaps(); };
$("marksend").onclick = async () => {
  const body = new FormData();
  body.append("action", "corners");
  body.append("image", pending, pending.name || "capture.jpg");
  body.append("corners", JSON.stringify(taps));
  $("marksend").disabled = true;
  $("marksend").textContent = "Solving\u2026";
  try {
    const r = await fetch(`/upload?t=${TOKEN}`, { method: "POST", body });
    show(await r.json());
  } catch (e) {
    show({ verdict: "FAILED", detail: "could not reach the laptop: " + e });
  }
  $("marksend").textContent = "Use these corners";
  $("marker").style.display = "none";
  taps = []; pending = null;
  refresh();
};

$("shot").onchange = async ev => {
  const file = ev.target.files[0];
  if (!file) return;
  if (action === "corners") {
    // Nothing is calibrated until the corners are confirmed, so the photograph
    // is shown here and held until then. In auto mode the laptop proposes the
    // dots first; they arrive already placed, and are dragged like any other.
    pending = file; taps = [];
    $("markimg").onload = paintTaps;
    $("markimg").src = URL.createObjectURL(file);
    $("marker").style.display = "block";
    $("result").style.display = "none";
    ev.target.value = "";
    $("marktitle").textContent = MODE === "auto" ? "Finding the display\u2026"
                                                 : "Tap the four corners of the display";
    if (MODE === "auto") {
      const body = new FormData();
      body.append("action", "propose");
      body.append("image", file, file.name || "capture.jpg");
      try {
        const r = await fetch(`/upload?t=${TOKEN}`, { method: "POST", body });
        const res = await r.json();
        if (res.proposal && pending === file) { taps = res.proposal.corners; }
        $("marktitle").textContent = res.proposal
          ? (res.proposal.confident ? "Found it \u2014 drag a dot if it's off"
                                    : "Check these \u2014 drag any dot that's off")
          : "Couldn't find it \u2014 tap the four corners";
      } catch (e) {
        $("marktitle").textContent = "Couldn't reach the laptop \u2014 tap the corners";
      }
      paintTaps();
    }
    return;
  }
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


def _bad_token_page(given: str) -> str:
    """Why the link did not work, on the device that is showing the problem.

    The cause is nearly always one of three things and the person holding the
    phone cannot see the terminal, so name all three and show what arrived.
    """
    shown = (given[:24] + "…") if len(given) > 24 else given
    received = (f"<p>This link carried <code>{_escape(shown)}</code>.</p>"
                if given else "<p>This link carried no token at all.</p>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Link not accepted</title>
<style>body{{margin:0;padding:24px;background:#0F1518;color:#E3EAE8;
font:16px/1.6 system-ui,-apple-system,sans-serif}}
h1{{font-size:19px;margin:0 0 12px}}code{{background:#1E272C;border:1px solid #2B373D;
border-radius:5px;padding:2px 6px;font-size:14px;word-break:break-all}}
li{{margin-bottom:8px}}p{{color:#9DADB5}}</style></head><body>
<h1>This link was not accepted</h1>
{received}
<p>Three things cause that:</p>
<ul>
<li><b>It was mistyped.</b> Scan the square the laptop printed instead of typing
    the address; that is what it is there for.</li>
<li><b>The server was restarted.</b> The token changes every run, so an address
    from an earlier one stops working. Use the one on screen now.</li>
<li><b>Part of the address was lost.</b> It has to keep the
    <code>?t=…</code> on the end.</li>
</ul>
</body></html>"""


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


class _Handler(BaseHTTPRequestHandler):
    server_version = "layoutval"
    sys_version = ""

    # -- plumbing -----------------------------------------------------------
    @property
    def session(self) -> CaptureSession:
        return self.server.session  # type: ignore[attr-defined]

    def _authorised(self, query: dict[str, list[str]]) -> bool:
        given = (query.get("t") or [""])[0]
        return token_matches(given, self.server.token)  # type: ignore[attr-defined]

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
            self._send(HTTPStatus.UNAUTHORIZED, _bad_token_page("").encode(),
                       "text/html; charset=utf-8")
            return
        if not self._authorised(query):
            self._send(HTTPStatus.FORBIDDEN,
                       _bad_token_page((query.get("t") or [""])[0]).encode(),
                       "text/html; charset=utf-8")
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

    def _set_mode(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 4096:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"detail": "too large"})
            return
        mode = _read_mode(self.rfile.read(length) if length else b"")
        try:
            self.session.set_mode(mode)
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            return
        if not self.server.quiet:  # type: ignore[attr-defined]
            print(f"  mode: {mode}")
        self._json(HTTPStatus.OK, self.session.status())

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if not self._authorised(parse_qs(url.query)):
            self._send(HTTPStatus.FORBIDDEN, b"bad token", "text/plain; charset=utf-8")
            return
        if url.path == "/mode":
            self._set_mode()
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
            action, image, corners = _parse_multipart(
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
            pixels = None
            if corners:
                # Fractions of the image, so they survive whatever size the
                # phone displayed the photograph at.
                shape = decode_upload(image).shape
                pixels = [[x * shape[1], y * shape[0]] for x, y in corners]
            record = self.session.handle(action, image, corners=pixels)
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


def _distort_points(points: np.ndarray, intrinsics: Any, new_k: np.ndarray) -> np.ndarray:
    """Undistorted-frame pixels back to the raw photograph's pixels.

    The inverse of the ``cv2.undistortPoints(..., P=new_K)`` that the corners
    route applies to taps, so a proposed dot that is accepted unchanged lands
    exactly where it was found.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    rays = np.column_stack([pts, np.ones(len(pts))]) @ np.linalg.inv(new_k).T
    raw, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                               intrinsics.K, intrinsics.dist)
    return raw.reshape(-1, 2)


def _read_mode(body: bytes) -> str:
    try:
        return str(json.loads(body.decode("utf-8") or "{}").get("mode", ""))
    except (ValueError, AttributeError):
        return ""


def _payload(record: CaptureRecord) -> dict[str, Any]:
    """What the phone needs to show, and no more."""
    out: dict[str, Any] = {
        "verdict": record.verdict,
        "detail": record.detail,
        "action": record.action,
    }
    if record.proposal:
        out["proposal"] = record.proposal
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


def _parse_multipart(
    body: bytes, content_type: str
) -> tuple[str, bytes, list[list[float]] | None]:
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
    action, image, corners = "", b"", None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if name == "action":
            action = payload.decode("utf-8", "replace").strip()
        elif name == "image":
            image = payload
        elif name == "corners":
            # Four tapped points, as fractions of the image. Fractions rather
            # than pixels because the phone scales the photograph to fit its
            # screen before anybody taps it.
            try:
                raw = json.loads(payload.decode("utf-8", "replace"))
                pts = [[float(x), float(y)] for x, y in raw][:4]
                corners = pts if len(pts) == 4 else None
            except (ValueError, TypeError):
                corners = None
    if not image:
        raise ValueError("no image in the upload")
    return action, image, corners


class CaptureServer(ThreadingHTTPServer):
    """The bench server.  One session, many phones, no state on the phone."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, session: CaptureSession, host: str, port: int,
                 *, token: str = "", quiet: bool = False) -> None:
        super().__init__((host, port), _Handler)
        self.session = session
        self.token = token or make_token()
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

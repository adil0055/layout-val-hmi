"""One button: Start camera, and the rest is done without asking.

The page streams the webcam's frames here and shows what this says. This runs
the whole sequence on a thread of its own and decides, from the frames
themselves, when the cluster has reacted to what was sent and which frames to
keep -- nobody picks a frame:

1. **Power cycle.** Ignition off, until the screen is dark and still; then on,
   recording the start-up: boot, the bulb check, the running screen.
2. **Locate the display.** What lit up between ignition off and on is the
   screen's content. The corners finder looks for the display's outline round
   it, and is held to containing it; where it finds none, the outline is
   taken round the content. When the resolution was not given, the display's
   size is worked out from its shape.
3. **Reference.** The running screen, at rest.
4. **Teach.** Every signal the cluster receives, state by state (see
   :mod:`layoutval.discover`), each shot once it has settled: a needle's
   sweep is waited out, a flashing lamp is shot across its flash. The bulb
   check says where every lamp is, those no signal lit included.
5. **Lamp check.** Every lamp state learnt, then all lamps together.
6. **Validate.** The layout at rest, against the reference.

Without CAN it is 2 (from the screen as it is), 3 and 6.

What "settled" means is :func:`watch`'s: a frame that matches one from a
moment before (a still screen, or a flash that has come round again), after
the screen has changed from how it was -- or, if it never changes, after long
enough to be sure it will not.
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

#: Width the screen is compared at, frame to frame. A 30 px lamp on a 1790 px
#: display is 8 px across here: enough to see it light, few enough pixels to
#: compare at the webcam's frame rate.
SMALL_W = 480
#: Fewest pixels (at SMALL_W) that make a change; fewer is noise.
CHANGE_PX = 6
#: After a state is sent: longest wait for the screen to react before taking it
#: that this state changes nothing.
REACT_S = 1.5
#: Frames that arrive this soon after a state is sent are of the state before;
#: the run measures its own, from the ignition off (see AutoRun.latency).
LATENCY_S = 0.12
LATENCY_MAX_S = 1.0
#: A frame that matches one from at least this long before is settled...
SETTLE_S = 0.25
#: ...as long as that one is no older than this: a flash comes round within it.
SETTLE_SPAN_S = 1.2
#: Longest wait for a change to settle (an animated element never does).
MAX_SETTLE_S = 4.0
#: Shots are taken across this long once settled, when the screen keeps
#: changing: two periods of a flash at 1.1 Hz and more, so that a flashing
#: lamp is seen to go on and off at least twice even at a few frames a second...
WINDOW_S = 1.8
#: ...unless nothing has changed for this long: no flash has halves longer...
STEADY_S = 0.7
#: ...over this many frames. At the few frames a second a webcam gets through
#: the page, two frames 0.7 s apart can both land in a flash's lit half, and
#: the flash was taught as a steady lamp: a real webcam's high beam, and the
#: simulated one's right indicator.
STEADY_FRAMES = 5
#: Shots kept of a state that flashes, and of one that does not.
SHOTS_FLASHING = 8
SHOTS_STEADY = 2
#: Ignition: how long the screen must stay still to have finished going dark,
#: or finished starting up, and the longest either may take.
DARK_STILL_S = 1.5
#: A bulb check can stand still for several seconds (3-5 s is usual); the
#: running screen is the one that stays still longer than that.
RUN_STILL_S = 6.0
POWER_MAX_S = 40.0
#: Back at rest, how long the screen is given to come all the way back: a
#: gauge is often damped, and slow to fall back.
RETURN_S = 8.0
#: After the camera's settings change, frames are let pass this long: a webcam
#: takes a few frames to move its exposure, and longer to move its lens.
CAMERA_SETTLE_S = 0.6
#: A page that has not put settings on the camera after this long will not.
CAMERA_ACK_S = 8.0
#: Frames of each exposure for a picture in two (camtune.fuse).
HDR_SHOTS = 3
#: A page that stops sending frames for this long has gone.
STALL_S = 20.0
#: Bulb check: a start-up frame with at least this share of the most compact
#: symbols lit (over the running screen) is a bulb-check frame.
BULB_SHARE = 0.7
BULB_MIN = 3
#: Start-up frames looked through for it, at most.
STARTUP_MAX = 120
#: Of the screen's content, the share the display's outline must contain.
CONTAIN_MIN = 0.97
#: Content's box is taken turned only when that is this much tighter.
TURNED_TIGHTER = 0.85
#: Content within this share of the picture's size of its edge runs off it...
EDGE_BAND = 0.01
#: ...when at least this share of what lit up is there, and this many pixels.
EDGE_SHARE = 0.002
EDGE_MIN_PX = 60
#: Common cluster resolutions: a display whose shape is within this of one is
#: taken to be it.
RESOLUTIONS = [(1920, 720), (1280, 480), (1440, 540), (2560, 960), (1920, 1080),
               (1280, 720), (1024, 600), (800, 480), (1920, 1200), (1280, 800),
               (2880, 1080), (1440, 480), (1888, 624)]
RESOLUTION_MATCH = 0.02


class Stopped(Exception):
    """Stop pressed, or the page went away."""


class CameraIgnored(Exception):
    """The page did not put the settings asked for on the camera."""


# -- frames -------------------------------------------------------------------

class FrameFeed:
    """The latest frame the page sent, for the run to wait on."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._seq = 0
        self._item: tuple[int, float, bytes, int] | None = None
        self.closed = False
        #: What the camera says it is set to, as the page last reported it.
        self.camera_report: dict | None = None

    def put(self, data: bytes, camera: int = 0, report: dict | None = None) -> int:
        """A frame, and which camera settings (by their number) it was taken with."""
        with self._cv:
            self._seq += 1
            self._item = (self._seq, time.monotonic(), data, int(camera))
            if report:
                self.camera_report = report
            self._cv.notify_all()
            return self._seq

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()

    def next(self, after: int, timeout: float = STALL_S) -> tuple[int, float, bytes, int]:
        """The first frame newer than ``after``: (sequence, arrival time, data, camera)."""
        end = time.monotonic() + timeout
        with self._cv:
            while not self.closed and (self._item is None or self._item[0] <= after):
                left = end - time.monotonic()
                if left <= 0:
                    raise Stopped("no frames from the page for "
                                  f"{timeout:.0f} s -- was it closed?")
                self._cv.wait(left)
            if self.closed:
                raise Stopped("stopped")
            return self._item


def small(img: np.ndarray, width: int = SMALL_W) -> np.ndarray:
    """A small grey copy, to compare frames by."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = width / g.shape[1]
    s = cv2.resize(g, (width, max(1, int(round(g.shape[0] * scale)))),
                   interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(s, (3, 3), 0)


#: A pixel has changed when it moved by more than this share of its
#: brightness, as well as by more than the noise. A backlight's PWM banding
#: through a camera is a gain on each row, a few percent either way, and on
#: white text that is tens of levels; a lamp lighting, a needle or a digit
#: moving changes a pixel by most of its brightness.
RELATIVE = 0.3


def _changed(a: np.ndarray, b: np.ndarray, level: float) -> np.ndarray:
    d = cv2.absdiff(a, b).astype(np.float32)
    top = np.maximum(a, b).astype(np.float32)
    return (d > level) & (d > RELATIVE * top)


def changed_px(a: np.ndarray, b: np.ndarray, level: float) -> int:
    return int(np.count_nonzero(_changed(a, b, level)))


#: Noise is never taken to be more than this: past it, something moved.
NOISE_MAX = 40.0


def noise_level(smalls: list[np.ndarray]) -> float:
    """The change between frames of a still screen, as a level to stay above."""
    if len(smalls) < 2:
        return 12.0
    # The largest difference between any two, not a high percentile: a
    # change is any CHANGE_PX pixels over the level, and the top 0.05 % of a
    # frame's noise is dozens of pixels. Differences a gain explains
    # (banding, see RELATIVE) are left out; they are not counted anyway.
    worst = 0.0
    for i, a in enumerate(smalls):
        for b in smalls[i + 1:]:
            d = cv2.absdiff(a, b).astype(np.float32)
            top = np.maximum(a, b).astype(np.float32)
            d[d <= RELATIVE * top] = 0
            worst = max(worst, float(d.max()))
    return float(min(NOISE_MAX, max(10.0, worst + 4.0)))


# -- settling -----------------------------------------------------------------

@dataclass
class Settled:
    shots: list[np.ndarray]           # what was kept, as the frames came
    last: np.ndarray                  # the last small frame, to compare the next state with
    changed: bool                     # the screen changed from how it was
    flashing: bool                    # it kept changing while shot
    settled: bool                     # it came to rest (or a flash came round)
    seconds: float
    changed_at: float | None = None   # the time of the first frame that had


def watch(frames: Iterator[tuple[float, np.ndarray, Any]], before: list[np.ndarray],
          level: float, *, react_s: float = REACT_S, window_s: float = WINDOW_S,
          settle_s: float = SETTLE_S, span_s: float = SETTLE_SPAN_S,
          max_settle_s: float = MAX_SETTLE_S, min_px: int = CHANGE_PX,
          steady_s: float = STEADY_S, steady_frames: int = STEADY_FRAMES,
          shots_flashing: int = SHOTS_FLASHING, shots_steady: int = SHOTS_STEADY,
          still_s: float | None = None) -> Settled:
    """Wait for the screen to react, to settle, and shoot it.

    ``frames`` yields (time, small frame, frame to keep), from after the state
    was sent. ``before`` is how the screen looked (one frame, or several of a
    flash): a change is a frame unlike all of them. ``still_s``, when given,
    asks for that long without any change instead of :data:`SETTLE_S`'s
    match -- a start-up has still phases of its own.
    """
    t0 = None
    changed = False
    changed_at = None
    settled = False
    hist: list[tuple[float, np.ndarray]] = []
    current = None
    for t, s, keep in frames:
        if t0 is None:
            t0 = t
        current = (t, s, keep)
        if not changed:
            if all(changed_px(s, b, level) >= min_px for b in before):
                changed = True
                changed_at = t
                hist = [(t, s)]
                continue
            if t - t0 >= react_s:
                break
            continue
        hist.append((t, s))
        if still_s is not None:
            run = [h for h in hist if t - h[0] <= still_s]
            if (hist[0][0] <= t - still_s
                    and all(changed_px(s, h, level) < min_px for _, h in run)):
                settled = True
                break
        elif any(settle_s <= t - th <= span_s and changed_px(s, h, level) < min_px
                 for th, h in hist):
            settled = True
            break
        if t - t0 >= max_settle_s:
            break
    if current is None:
        raise Stopped("no frames")
    # Shots across the window, from the frame it settled on; a still screen
    # is done sooner.
    window = [current]
    first = current[1]
    flashing = False
    for t, s, keep in frames:
        window.append((t, s, keep))
        flashing = flashing or changed_px(first, s, level) >= min_px
        span = t - window[0][0]
        if span >= window_s or (span >= steady_s and len(window) >= steady_frames
                                and not flashing):
            break
    want = shots_flashing if flashing else shots_steady
    if len(window) > want:
        idx = np.linspace(0, len(window) - 1, want).round().astype(int)
        picked = [window[i] for i in sorted(set(idx.tolist()))]
    else:
        picked = window
    return Settled(shots=[k for _, _, k in picked], last=window[-1][1], changed=changed,
                   changed_at=changed_at,
                   flashing=flashing, settled=settled or not changed,
                   seconds=window[-1][0] - (t0 or window[-1][0]))


# -- finding the display -------------------------------------------------------

def lit_region(dark: np.ndarray, on: np.ndarray, level: float = 18.0) -> np.ndarray:
    """Where the screen lit up between ``dark`` and ``on`` (camera frames), as a mask.

    Brighter, not just different: a webcam still on automatic brightens the
    whole room when the screen goes dark, and the room changing that way is
    not the screen lighting up.
    """
    a = cv2.GaussianBlur(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(on, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = (cv2.subtract(b, a) > level).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    # Specks far from the rest (a reflection that changed) are not the screen.
    keep = np.zeros(n, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep[1:] = areas >= max(20, 0.0005 * areas.sum())
    return keep[labels]


def ordered(points: np.ndarray) -> np.ndarray:
    """Four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, np.float64).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def contains(corners: np.ndarray, mask: np.ndarray) -> float:
    """The share of ``mask`` inside the quadrilateral ``corners``."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    poly = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(poly, np.round(ordered(corners)).astype(np.int32), 1)
    return float((mask & poly.astype(bool)).sum()) / total


def cut_edges(mask: np.ndarray, valid: np.ndarray | None = None) -> list[str]:
    """The edges of the camera's picture that what lit up runs off.

    ``valid`` is the part of the picture there is (undistorting leaves a
    border with nothing in it). Content in a band along its edge -- more than
    a speck of it -- is content the camera does not see all of.
    """
    h, w = mask.shape
    total = int(mask.sum())
    if total == 0:
        return []
    if valid is None:
        valid = np.ones(mask.shape, bool)
    k = max(2, round(EDGE_BAND * min(h, w)))
    inner = cv2.erode(valid.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    ys, xs = np.nonzero(mask & valid & ~inner)
    if len(xs) < max(EDGE_MIN_PX, EDGE_SHARE * total):
        return []
    # Each to the side of the picture it is nearest.
    side = np.argmin(np.stack([ys, h - 1 - ys, xs, w - 1 - xs]), axis=0)
    counts = np.bincount(side, minlength=4)
    names = ("top", "bottom", "left", "right")
    return [names[i] for i in range(4) if counts[i] >= max(EDGE_MIN_PX / 2, 0.25 * len(xs))]


def content_corners(mask: np.ndarray, margin: float = 0.01) -> np.ndarray:
    """A rectangle round the content, a little larger: upright, unless the
    content is clearly turned.

    The tightest rectangle round a cluster's content is turned whenever the
    content is not symmetric -- a readout below one gauge -- and the picture
    taken through it came out turned and skewed though the camera was level.
    So the upright box is taken unless the turned one is much the tighter
    (a camera rolled well over).
    """
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    (cx, cy), (w, h), a = rect
    bx, by, bw, bh = cv2.boundingRect(hull)
    if w * h >= TURNED_TIGHTER * bw * bh:
        rect = ((bx + bw / 2.0, by + bh / 2.0), (float(bw), float(bh)), 0.0)
        (cx, cy), (w, h), a = rect
    rect = ((cx, cy), (w * (1 + 2 * margin), h * (1 + 2 * margin)), a)
    return ordered(cv2.boxPoints(rect))


def content_size(corners: np.ndarray) -> tuple[int, int]:
    """A box round the content as display pixels: one per camera pixel.

    The box is not the display, so its shape says nothing of the display's
    resolution, and stretching it to a common one (16:9 made 1920 x 1080 of
    a 2.6:1 cluster's content) only distorts it.
    """
    c = ordered(corners)
    w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
    h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
    return int(round(w / 2.0)) * 2, int(round(h / 2.0)) * 2


def estimate_size(corners: np.ndarray, image_size: tuple[int, int]) -> tuple[tuple[int, int], str]:
    """The display's resolution from its shape: a common one it matches, or one of its shape.

    Returns ((width, height), how it was decided).
    """
    from layoutval.displayfind import rectangle_aspect

    aspect = rectangle_aspect(ordered(corners), image_size)
    for w, h in RESOLUTIONS:
        if abs((w / h) / aspect - 1.0) <= RESOLUTION_MATCH:
            return (w, h), f"its shape ({aspect:.3f}) is that of {w} x {h}"
    # As many display pixels across as the camera sees, so nothing is lost.
    c = ordered(corners)
    across = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
    w = int(round(max(640.0, min(2560.0, across)) / 8.0) * 8)
    h = int(round(w / aspect / 2.0) * 2)
    return (w, h), f"its shape ({aspect:.3f}); no common resolution has it"


def bulb_frames(run: np.ndarray, frames: list[np.ndarray], level: float) -> tuple[list[int], list[int]]:
    """Which start-up frames (display space) are the bulb check: (indices, symbol counts).

    The bulb check is the start-up's frames with the most compact symbols lit
    where the running screen is dark -- every lamp at once. The boot logo's
    letters are a few; a bulb check is every lamp.
    """
    from layoutval import lamps as L

    dark = run.max(axis=2) < 70
    counts = []
    for f in frames:
        delta = L.lit_delta(run, f)
        bs = L.blobs(delta, level)
        n = 0
        for b in bs:
            x, y, w, h = b.box
            if (dark & b.mask).sum() < 0.6 * b.area:
                continue
            if not (0.33 <= w / max(h, 1) <= 3.0) or max(w, h) > 0.15 * run.shape[1]:
                continue
            n += 1
        counts.append(n)
    if not counts or max(counts) < BULB_MIN:
        return [], counts
    top = max(counts)
    return [i for i, n in enumerate(counts) if n >= BULB_SHARE * top], counts


# -- the run -----------------------------------------------------------------

@dataclass
class Progress:
    phase: str = "starting"
    detail: str = ""
    step: int = 0
    steps: int = 0
    done: bool = False
    record: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)
    #: Settings for the page to put on the camera: {"seq": n, "set": {...}}.
    camera: dict[str, Any] | None = None
    #: What the run waits on the person for: {"what": "corners", ...}.
    ask: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "detail": self.detail, "step": self.step,
                "steps": self.steps, "done": self.done, "record": self.record,
                "camera": self.camera, "ask": self.ask}


class AutoRun:
    """The whole sequence, for one session, on its own thread."""

    def __init__(self, session, *, teach: bool | None = None, keep_shots: bool = False,
                 camera: dict | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.s = session
        #: What the page's camera can be set to and is set to: {"caps", "settings"}.
        self.camera = camera or {}
        self._camera_seq = 0
        self._camera_seen = 0
        #: Teach (learn the cluster: display, reference, lamps), or test a
        #: cluster against what was taught. Taught already, a run tests.
        self.teach = (session.lamp_lib is None or session.reference is None) \
            if teach is None else teach
        self.feed = FrameFeed()
        self.progress = Progress()
        self.keep_shots = keep_shots
        self._log = log or (lambda m: None)
        self._seq = 0
        self._undistorter = None
        #: The pose this run measures through: the calibration's, or where a
        #: test finds the screen's outline now (a webcam nudged since teaching).
        self._H: np.ndarray | None = None
        #: What lit up at ignition on, in the camera's picture: the display's
        #: content, and not a lamp in the room beside it.
        self._lit: np.ndarray | None = None
        #: The display's corners in the camera's picture, and how they were
        #: had, from the first look: the second, after the camera is set up,
        #: uses them again rather than asking twice.
        self._corners: tuple[np.ndarray, str] | None = None
        self._answer: Any = None
        self._answered = threading.Event()
        #: Each lamp as the lamp check saw it lit, for the element pictures.
        self._lamp_pics: dict[str, list[dict[str, Any]]] = {}
        self._thread: threading.Thread | None = None
        self.level = 12.0
        #: From a state sent to the first frame showing it: the bus, the
        #: cluster, the camera and the page. Measured at the ignition off.
        self.latency = LATENCY_S
        self.stamp = datetime.now().strftime("%H%M%S")
        self.folder = Path(session.out_dir) / f"auto-{self.stamp}"
        self.summary: dict[str, Any] = {"started": datetime.now().isoformat(timespec="seconds")}

    # -- plumbing --------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="layoutval-auto", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.feed.close()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _say(self, phase: str, detail: str, step: int = 0, steps: int = 0) -> None:
        p = self.progress
        p.phase, p.detail, p.step, p.steps = phase, detail, step, steps
        self._log(f"auto: {phase}: {detail}")

    def _frame(self) -> tuple[float, np.ndarray, bytes]:
        from layoutval.server import decode_upload

        seq, t, data, camera = self.feed.next(self._seq)
        self._seq = seq
        self._camera_seen = camera
        return t, decode_upload(data), data

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        cal = self.s.calibration
        if cal is None or cal.intrinsics is None:
            return frame
        if self._undistorter is None:
            from layoutval.calibration import Undistorter
            self._undistorter = Undistorter(cal.intrinsics)
        return self._undistorter(frame)

    def _display(self, frame: np.ndarray) -> np.ndarray:
        return self.s.calibration.geometry.rectify(self._undistort(frame), self._H)

    def _camera_frames(self, since: float) -> Iterator[tuple[float, np.ndarray, tuple]]:
        """(time, small, (frame, data)) of whole camera frames, from ``since`` on."""
        while True:
            t, f, data = self._frame()
            if t < since:
                continue
            yield t, small(f), (f, data)

    def _display_frames(self, since: float) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
        """(time, small, display-space frame), from ``since`` on."""
        while True:
            t, f, _ = self._frame()
            if t < since:
                continue
            d = self._display(f)
            yield t, small(d), d

    def _still(self, count: int = 4, timeout: float = POWER_MAX_S) -> list[np.ndarray]:
        """``count`` small frames in a row that agree: the picture is still.

        The cluster may be part way through starting when the camera opens;
        its noise is measured once it has finished.
        """
        frames: list[np.ndarray] = []
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            frames.append(small(self._frame()[1]))
            frames = frames[-count:]
            if len(frames) == count:
                level = noise_level(frames)
                if level < NOISE_MAX and all(changed_px(frames[-1], f, level) < CHANGE_PX
                                             for f in frames[:-1]):
                    return frames
        return frames

    # -- the camera -------------------------------------------------------------

    def _controllable(self) -> bool:
        from layoutval.camtune import controls
        return bool(controls(self.camera.get("caps") or {}))

    def _camera_look(self, settings: dict, shots: int = 2) -> list[np.ndarray]:
        """Have the page put ``settings`` on the camera; frames taken with them.

        Frames come tagged with the number of the settings they were taken
        with, and the first moments after a change are let pass: a webcam
        takes a few frames to move its exposure, and longer to move its lens.
        """
        self._camera_seq += 1
        n = self._camera_seq
        self.progress.camera = {"seq": n, "set": dict(settings)}
        asked = time.monotonic()
        since = None
        frames: list[np.ndarray] = []
        while True:
            t, f, _ = self._frame()
            if self._camera_seen < n:
                if time.monotonic() - asked > CAMERA_ACK_S:
                    raise CameraIgnored("the page did not set the camera")
                continue
            if since is None:
                since = t
            if t - since < CAMERA_SETTLE_S:
                continue
            frames.append(f)
            if len(frames) >= shots:
                return frames

    def _tune(self) -> str:
        """Set the camera up for the display (layoutval.camtune); what it came to."""
        from layoutval.camtune import Tuner, controls

        caps = self.camera.get("caps") or {}
        can = controls(caps)
        if not can:
            self.summary["camera"] = {"tuned": False,
                                      "why": "nothing on it can be set from this browser"}
            return ""
        self._say("camera", "setting the camera up for the display: " + ", ".join(can))
        content = None
        if self._lit is not None:
            # What lit up, in display space, with a margin for the edges a
            # focus measure needs.
            g = self.s.calibration.geometry
            content = g.rectify((self._lit * 255).astype(np.uint8), self._H) > 127
            k = max(3, round(0.006 * content.shape[1]))
            content = cv2.dilate(content.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        tuner = Tuner(caps, self.camera.get("settings") or {}, self._camera_look,
                      region=self._display, log=self._log, mask=content)
        try:
            final = tuner.run()
        except CameraIgnored as exc:
            self.summary["camera"] = {"tuned": False, "why": str(exc)}
            return ""
        report = self.feed.camera_report or {}
        # What the camera says it took, where it says: steps round values.
        final = {k: report.get(k, v) for k, v in final.items()}
        self.s.camera_settings = final
        self.summary["camera"] = {"tuned": True, "settings": final, "steps": tuner.steps}
        try:
            (self.folder / "camera.json").write_text(json.dumps(
                {"settings": final, "steps": tuner.steps, "capabilities": caps}, indent=2,
                default=str))
        except OSError:
            pass
        bits = []
        if "exposureTime" in final:
            bits.append(f"exposure {final['exposureTime']:g}")
        elif "exposureCompensation" in final:
            bits.append(f"exposure {final['exposureCompensation']:+g} EV")
        if "colorTemperature" in final:
            bits.append(f"white balance {float(final['colorTemperature']):.0f} K")
        if "focusDistance" in final:
            bits.append(f"focus {float(final['focusDistance']):g}")
        if "sharpness" in final:
            bits.append(f"sharpening {final['sharpness']:g}")
        if "iso" in final:
            bits.append(f"gain ISO {final['iso']:g}")
        if "brightness" in final and final.get("brightness") != (self.camera.get("settings") or {}).get("brightness"):
            bits.append(f"brightness {float(final['brightness']):g}")
        return "set for the display and held: " + ", ".join(bits)

    def _hdr(self) -> np.ndarray | None:
        """The screen as it is now, in one picture from two exposures (camtune.fuse).

        None where the camera was not set up, or cannot go longer. The camera
        is left at the exposure set for the display.
        """
        from layoutval.camtune import bracket, fuse

        s = self.s
        if not s.camera_settings or not self._controllable():
            return None
        longer = bracket(self.camera.get("caps") or {}, s.camera_settings)
        if longer is None:
            return None
        try:
            short = self._camera_look(s.camera_settings, shots=HDR_SHOTS)
            long = self._camera_look(longer, shots=HDR_SHOTS)
            self._camera_look(s.camera_settings, shots=1)
        except CameraIgnored:
            return None

        def median(fs):
            return np.median(np.stack(fs), axis=0).astype(np.uint8)

        return fuse(median(short), median(long))

    def _send(self, values: dict[str, Any]) -> float:
        self.s.can.apply(values)
        return time.monotonic() + self.latency

    # -- the sequence ----------------------------------------------------------

    def _main(self) -> None:
        try:
            rec = self._run()
        except Stopped as exc:
            rec = self._finish("FAILED", f"Stopped: {exc}. Nothing further was done.")
        except Exception as exc:          # a bench tool: say what broke, keep serving
            self._log(traceback.format_exc())
            rec = self._finish("FAILED", f"{type(exc).__name__}: {exc}")
        finally:
            if self.s.can is not None:
                try:
                    self.s.can.quiet(False)
                    self.s.can.rest()
                except Exception:
                    pass
        self.progress.record = rec
        self.progress.done = True
        self.feed.close()

    def _finish(self, verdict: str, detail: str, rows: list | None = None,
                overlay: str | None = None) -> dict[str, Any]:
        self.summary.update({"verdict": verdict, "detail": detail,
                             "finished": datetime.now().isoformat(timespec="seconds")})
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "auto.json").write_text(json.dumps(self.summary, indent=2, default=str))
        except OSError:
            pass
        out = {"verdict": verdict, "detail": detail, "action": "auto", "rows": rows or []}
        if overlay:
            out["overlay"] = overlay
        return out

    def _run(self) -> dict[str, Any]:
        s = self.s
        self._started = time.monotonic()
        s.steady_camera = True              # a webcam on a stand, not a hand
        self.folder.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        can = s.can is not None
        taught = getattr(s, "discovery", None) is not None or s.plan is not None
        nothing_taught = not s.is_calibrated or s.reference is None or (
            can and taught and s.lamp_lib is None)
        if self.teach is False and nothing_taught:
            return self._finish("FAILED", "Nothing is taught yet: press Teach first, then Test.")
        teaching = self.teach or nothing_taught
        self.summary["mode"] = "teach" if teaching else "test"

        # The camera as it was set up when taught: a test seen at another
        # exposure would differ from its reference everywhere.
        if not teaching and s.camera_settings and self._controllable():
            self._say("camera", "setting the camera as it was when taught")
            try:
                self._camera_look(s.camera_settings)
                self.summary["camera"] = {"settings": s.camera_settings, "tuned": False}
            except CameraIgnored as exc:
                self.summary["camera"] = {"tuned": False, "why": str(exc)}

        # 1-2: the display.
        if can:
            dark, run_frames, startup = self._power_cycle()
        else:
            dark, startup = None, []
            run_frames = self._still_frames()
        if teaching:
            where = self._locate(dark, run_frames, startup)
            tuned = self._tune()
            if tuned:
                # Everything from here on with the camera as it now is: the
                # start-up and the display again (focus moves the picture a
                # little), at the exposure the reference will have.
                rows.append({"id": "Camera", "verdict": "PASS", "detail": tuned})
                if can:
                    dark, run_frames, startup = self._power_cycle()
                else:
                    run_frames = self._still_frames()
                where = self._locate(dark, run_frames, startup)
            rows.append({"id": "Display", "verdict": "PASS", "detail": where})
        elif s.reference_camera is not None:
            frame = run_frames[-1][0]
            if s.reference_hdr:
                # Compared with the reference, taken the same way.
                hdr = self._hdr()
                frame = hdr if hdr is not None else frame
            with s.lock:
                held = s._held_pose(self._undistort(frame))
            if held is not None:
                self._H = held[0]
                self.summary["camera_moved_px"] = round(float(held[1]), 2)
        rest = [self._display(f) for f, _ in run_frames[-3:]]
        self.level = noise_level([small(d) for d in rest])
        bulb = self._bulb(run_frames, startup) if can and startup else []

        if teaching:
            # 3: reference.
            self._say("reference", "taking the reference: the screen at rest")
            raw = np.median(np.stack([f for f, _ in run_frames[-3:]]), axis=0).astype(np.uint8)
            # In two exposures where the camera can: the white text crisp and
            # the display's dim parts (dial faces, rings, minor ticks) seen.
            hdr = self._hdr()
            self.summary["hdr"] = hdr is not None
            with s.lock:
                ref = s._record(s._reference(raw if hdr is None else hdr,
                                             f"{self.stamp}-reference", hdr=hdr is not None))
            if ref.verdict == "FAILED":
                return self._finish("FAILED", f"Reference: {ref.detail}", rows)
            rows.append({"id": "Reference", "verdict": "PASS",
                         "detail": f"{len(s.profile) if s.profile else 0} elements on the rest screen"
                                   + ("; in two exposures, so the dim parts are seen too"
                                      if hdr is not None else "")})
            # 4: teach.
            if can and taught:
                lib, teach_rows = self._teach(rest, bulb)
                rows += teach_rows
                if bulb:
                    # Each lamp as the bulb check showed it, for its picture.
                    pic = np.max(np.stack(bulb), axis=0)
                    for lamp in lib.lamps:
                        self._lamp_pics.setdefault(lamp.name, []).append(
                            self._lamp_picture(pic, lamp.box, "bulb check", "PASS", lamp))
            # Taught: testing is for when it is asked for (Test).
            sheet = None
            try:
                sheet = self._element_pictures(None, rows)
            except Exception:
                self._log(traceback.format_exc())
            worst = _worst([r["verdict"] for r in rows])
            n_lamps = len(s.lamp_lib.lamps) if s.lamp_lib is not None else 0
            detail = (f"Taught in {time.monotonic() - self._started:.0f} s: "
                      f"{len(s.profile) if s.profile else 0} elements"
                      + (f", {n_lamps} lamps" if can and taught else "")
                      + f". Press Test to check a cluster. Everything is in {self.folder.name}/.")
            order = {"FAIL": 0, "FAILED": 0, "REVIEW": 1}
            rows.sort(key=lambda r: order.get(r["verdict"], 2))
            out = self._finish(worst if worst != "PASS" else "TAUGHT", detail, rows[:120])
            if sheet:
                out["elements"] = sheet
            return out
        elif s.lamp_lib is not None and can and (
                bulb or any(lamp.slot_check is not None for lamp in s.lamp_lib.lamps)):
            # Taught with a bulb check, tested without one seen is a fault too.
            rows += self._bulb_test(s.lamp_lib, bulb, rest)

        # 5: lamp check.
        lib = s.lamp_lib
        if can and lib is not None and lib.lamps:
            rows += self._check(lib)

        # 6: validate.
        self._say("validate", "validating the layout at rest")
        since = self._send({}) if can else time.monotonic()
        got = watch(self._camera_frames(since), [small(f) for f, _ in run_frames[-1:]],
                    self.level, react_s=1.0, shots_steady=3)
        frames = [f for f, _ in got.shots]
        raw = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 1 else frames[0]
        shot = raw
        if s.reference_hdr:
            hdr = self._hdr()
            if hdr is not None:
                shot = hdr
            else:
                rows.append({"id": "Camera", "verdict": "REVIEW",
                             "detail": "the reference was taken in two exposures and the camera "
                                       "would not take the second now: the layout is compared "
                                       "with a picture taken another way"})
        with s.lock:
            # Lamps lit at rest are looked for in the plain picture: the lamp
            # library's rest screen was taken so.
            val = s._record(s._validate(shot, f"{self.stamp}-validate", lamps_from=raw))
        from layoutval.server import _payload
        payload = _payload(val)
        verdict = val.verdict
        self.summary["layout"] = payload.get("detail", val.detail)
        rows.append({"id": "Layout", "verdict": verdict if verdict in ("PASS", "REVIEW", "FAIL")
                     else "REVIEW", "detail": payload.get("detail", val.detail)})
        rows += [r for r in payload.get("rows", [])]
        rows += [{"id": "Layout", "verdict": "REVIEW", "detail": f} for f in payload.get("flags", [])]
        worst = _worst([r["verdict"] for r in rows])
        detail = (f"{'Taught and checked' if teaching else 'Tested'} in "
                  f"{time.monotonic() - self._started:.0f} s. Layout {verdict}"
                  + (f"; lamps {self.summary['lamp_check']}" if "lamp_check" in self.summary else "")
                  + f". Everything is in {self.folder.name}/.")
        # What failed first, then what is for review, then the rest.
        order = {"FAIL": 0, "FAILED": 0, "REVIEW": 1}
        rows.sort(key=lambda r: order.get(r["verdict"], 2))
        sheet = None
        try:
            sheet = self._element_pictures(val, rows)
        except Exception:                    # pictures are for people; never the verdict
            self._log(traceback.format_exc())
        out = self._finish(worst, detail, rows[:120], payload.get("overlay"))
        if sheet:
            out["elements"] = sheet
        return out

    # -- 1. power ---------------------------------------------------------------

    def _power_cycle(self) -> tuple[np.ndarray, list[tuple[np.ndarray, bytes]],
                                    list[tuple[float, bytes]]]:
        """Ignition off until dark and still, then on until running and still.

        Returns the dark frame, the running screen's last frames (frame, data)
        and every start-up frame's (time, data).
        """
        can = self.s.can
        # How much a still screen changes frame to frame through this camera
        # (noise, a backlight's flicker), from the screen as it is now.
        self._say("power", "waiting for a still picture")
        first = self._still()
        level = noise_level(first)
        before = first[-1:]
        self._say("power", "ignition off: waiting for the screen to go dark")
        sent = time.monotonic()
        can.ignition(False)
        off = watch(self._camera_frames(time.monotonic() + LATENCY_S), before, level,
                    react_s=6.0, still_s=DARK_STILL_S, max_settle_s=POWER_MAX_S,
                    window_s=0.4, shots_steady=3)
        if not off.changed:
            # Some clusters stay lit with the ignition off while the bus is
            # awake: let the bus sleep as well.
            self._say("power", "nothing changed with the ignition off: letting the bus sleep")
            can.quiet(True)
            off = watch(self._camera_frames(time.monotonic() + LATENCY_S), before, level,
                        react_s=8.0, still_s=DARK_STILL_S, max_settle_s=POWER_MAX_S,
                        window_s=0.4, shots_steady=3)
        if not off.changed:
            raise Stopped("the screen did not change with the ignition off and the bus "
                          "asleep -- is the camera on the cluster, and the cluster on this bus?")
        dark = off.shots[-1][0]
        delays = [off.changed_at - sent] if off.changed_at is not None else []
        self._say("power", "ignition on: recording the start-up")
        startup: list[tuple[float, bytes]] = []

        def recorded(frames):
            # Ten a second is plenty to catch a bulb check of a second or more.
            for t, s_, (f, data) in frames:
                if not startup or t - startup[-1][0] >= 0.1:
                    startup.append((t, data))
                yield t, s_, (f, data)

        can.quiet(False)
        sent = time.monotonic()
        can.ignition(True)
        on = watch(recorded(self._camera_frames(time.monotonic() + LATENCY_S)),
                   [small(dark)], level, react_s=10.0, still_s=RUN_STILL_S,
                   max_settle_s=POWER_MAX_S, window_s=0.5, shots_steady=3)
        if not on.changed:
            raise Stopped("the screen stayed dark with the ignition on")
        if on.changed_at is not None:
            delays.append(on.changed_at - sent)
        if delays:
            # The longer of the two, whole: a frame arrives only every so
            # often, so one reading can come out short. Too long costs a
            # little time; too short mixes frames of the state before into
            # the next one.
            self.latency = float(np.clip(max(delays), LATENCY_S, LATENCY_MAX_S))
        self.summary["power"] = {"startup_s": round(on.seconds, 1), "startup_frames": len(startup),
                                 "came_to_rest": on.settled, "noise_level": round(level, 1),
                                 "latency_s": round(self.latency, 2)}
        return dark, list(on.shots), startup

    def _still_frames(self) -> list[tuple[np.ndarray, bytes]]:
        """Without CAN: the screen as it is, once it is still."""
        self._say("locate", "waiting for a still picture")
        first = self._still()
        got = watch(self._camera_frames(time.monotonic()), first[-1:], noise_level(first),
                    react_s=0.0, window_s=0.6, shots_steady=3)
        return list(got.shots)

    # -- 2. locate -------------------------------------------------------------

    def _locate(self, dark: np.ndarray | None, run_frames, startup=()) -> str:
        from layoutval.calibration import homography_from_marked_corners
        from layoutval.displayfind import propose_display_corners

        s = self.s
        self._say("locate", "finding the display")
        on = np.median(np.stack([f for f, _ in run_frames]), axis=0).astype(np.uint8)
        on_u = self._undistort(on)
        lit = None
        off: list[str] = []
        mode = s.corner_check
        if dark is not None:
            # Everything that lit at any time while starting, not just the
            # running screen: the bulb check lights lamps the running screen
            # leaves dark, and on a black screen they can sit above anything
            # else -- the outline round the gauges alone cut the BMW design's
            # top row of lamps off.
            dark_u = self._undistort(dark)
            lit = lit_region(dark_u, on_u)
            from layoutval.server import decode_upload
            picks = np.linspace(0, len(startup) - 1, min(24, len(startup))).round().astype(int) \
                if len(startup) else []
            for i in sorted(set(int(k) for k in picks)):
                lit |= lit_region(dark_u, self._undistort(decode_upload(startup[i][1])))
            self._lit = lit
            valid = self._undistort(np.full(on.shape[:2], 255, np.uint8)) > 0
            off = cut_edges(lit, valid)
            if off and mode == "auto":
                cv2.imwrite(str(self.folder / "lit-at-ignition.png"), (lit * 255).astype(np.uint8))
                cv2.imwrite(str(self.folder / "camera.jpg"), on_u)
                sides = " and ".join(off)
                self.summary["display"] = {"runs_off": off}
                raise Stopped(f"the display runs off the {sides} of the camera's picture, so "
                              "part of it is not seen. Aim the camera so all of the display is "
                              "in view, with a little room round it, and press Start again")
        if self._corners is not None:
            # As they were found -- and checked -- at the first look: the
            # camera has not moved since.
            corners, how = self._corners
        else:
            corners, how = None, ""
            if mode != "manual":
                aspect = (s.corner_display_size[0] / s.corner_display_size[1]) \
                    if s.size_set else None
                proposal = propose_display_corners(on_u, aspect=aspect)
                if proposal is not None:
                    corners, how = ordered(proposal.corners), "its outline"
                    if lit is not None and lit.any():
                        share = contains(corners, lit)
                        if share < CONTAIN_MIN:
                            self._log(f"auto: the outline found holds {share:.0%} of what lit "
                                      "up; taking the outline round the content instead")
                            corners = None
                if corners is None and lit is not None and lit.any():
                    corners, how = content_corners(lit), "what lit up at ignition on"
            if mode in ("check", "manual"):
                corners, how = self._ask_corners(on_u, corners, how, off)
            elif corners is None:
                raise Stopped("could not find the display in the picture")
            self._corners = (corners, how)
        outline = how in ("its outline", "the corners you marked")
        size_note = ""
        # Estimated again on the second look, with the camera set up: the
        # first was through a camera on automatic.
        if not s.size_set or "display_size_estimated" in self.summary:
            if outline:
                size, why = estimate_size(corners, (on_u.shape[1], on_u.shape[0]))
            else:
                size = content_size(corners)
                why = "the box round what lit up, as the camera sees it"
            s.set_size(*size)
            size_note = f"; taken to be {size[0]} x {size[1]}: {why}"
            self.summary["display_size_estimated"] = list(size)
        with s.lock:
            try:
                # Corners onto the display's edge -- where there is one: a box
                # round the content has none, and its corners would be pulled
                # onto the ticks and numerals next to them.
                geometry = homography_from_marked_corners(
                    on_u, corners, display_size=s.corner_display_size, refine=outline,
                    inset_px=s.display_inset_px)
            except RuntimeError:
                geometry = homography_from_marked_corners(
                    on_u, corners, display_size=s.corner_display_size, refine=False,
                    inset_px=s.display_inset_px)
            s._commit_geometry(geometry, f"automatic: the display from {how}")
        from layoutval.calibration import draw_aperture
        cv2.imwrite(str(self.folder / "display.jpg"), draw_aperture(on_u, geometry.corners))
        if lit is not None:
            cv2.imwrite(str(self.folder / "lit-at-ignition.png"), (lit * 255).astype(np.uint8))
        w, h = s.display_size
        ratio = geometry.sampling_ratio()
        self.summary["display"] = {"from": how, "size": [w, h], "sampling_ratio": round(ratio, 2),
                                   "corners": np.round(geometry.corners, 1).tolist()}
        return (f"found from {how}, {w} x {h}{size_note}; "
                f"{ratio:.1f} camera px per display px")

    def _ask_corners(self, img: np.ndarray, corners, how: str,
                     off: list[str]) -> tuple[np.ndarray, str]:
        """The display's corners as the person at the page has them: the ones
        found, checked and dragged where they are off, or tapped from scratch."""
        h, w = img.shape[:2]
        self.folder.mkdir(parents=True, exist_ok=True)
        name = f"{self.folder.name}/corners.jpg"
        cv2.imwrite(str(self.s.out_dir / name), img)
        if corners is None:
            note = "The display was not found: tap its four corners."
        elif how == "its outline":
            note = "Found by its outline. Drag any corner that is off."
        else:
            note = ("No outline found; this is a box round what lit up. Drag the "
                    "corners onto the display's own corners if you can see them.")
        if off:
            note += (f" What lit up runs off the {' and '.join(off)} of the picture: "
                     "part of the display may be out of view.")
        self._answer, self._answered = None, threading.Event()
        self.progress.ask = {
            "what": "corners", "image": name, "note": note,
            "corners": None if corners is None
            else (np.asarray(corners, float) / [w, h]).round(5).tolist()}
        self._say("locate", "check the display's corners, then Use these corners")
        while not self._answered.wait(0.3):
            if self.feed.closed:
                raise Stopped("stopped while the corners were being checked")
        self.progress.ask = None
        pts = ordered(np.asarray(self._answer, float).reshape(4, 2) * [w, h])
        moved = corners is None or float(np.abs(pts - ordered(corners)).max()) > 2.0
        self.summary["corners"] = "marked" if moved else "checked"
        return pts, ("the corners you marked" if moved else how)

    def answer_corners(self, corners) -> None:
        """The corners from the page: four (x, y), each 0-1 across the picture."""
        pts = np.asarray(corners, float)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)) or pts.min() < 0 or pts.max() > 1:
            raise ValueError("four corners, each as x and y between 0 and 1")
        if self.progress.ask is None or self.progress.ask.get("what") != "corners":
            raise ValueError("the run is not waiting for corners")
        self._answer = pts.tolist()
        self._answered.set()

    # -- 4. teach ----------------------------------------------------------------

    def _bulb(self, run_frames, startup) -> list[np.ndarray]:
        from layoutval.server import decode_upload
        from layoutval import lamps as L

        self._say("teach", "finding the bulb check in the start-up")
        run = np.median(np.stack([self._display(f) for f, _ in run_frames]), axis=0).astype(np.uint8)
        recorded = len(startup)
        span = round(startup[-1][0] - startup[0][0], 1) if len(startup) > 1 else 0.0
        if len(startup) > STARTUP_MAX:
            idx = np.linspace(0, len(startup) - 1, STARTUP_MAX).round().astype(int)
            startup = [startup[i] for i in idx]
        frames = [self._display(decode_upload(data)) for _, data in startup]
        floor, _ = L.noise_floor([self._display(f) for f, _ in run_frames])
        idx, counts = bulb_frames(run, frames, L._threshold(floor, L.LIT_MIN))
        self.summary["bulb_check"] = {
            "frames": len(idx), "symbols": max(counts) if counts else 0, "counts": counts,
            "startup_frames": recorded, "startup_s": span}
        picked = [frames[i] for i in idx]
        if picked:
            cv2.imwrite(str(self.folder / "bulb-check.png"),
                        np.max(np.stack(picked), axis=0))
        return picked

    def _teach(self, rest: list[np.ndarray], bulb: list[np.ndarray]):
        import queue

        from layoutval import lamps as L
        from layoutval.server import _ShotFiles

        s = self.s
        disc = getattr(s, "discovery", None)
        steps = disc.steps() if disc is not None else s.plan.steps()
        shots_dir = self.folder / "teach"
        files: dict[str, dict[int, Path]] = {}
        captured: dict[str, list[np.ndarray]] = {"rest": rest}

        def keep(step_id: str, frames: list[np.ndarray]) -> None:
            if disc is not None and not self.keep_shots:
                captured[step_id] = frames
                return
            shots_dir.mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(frames):
                path = shots_dir / f"{step_id}-{k}.png"
                cv2.imwrite(str(path), f, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                files.setdefault(step_id, {})[k] = path
            captured[step_id] = frames

        # Each signal is looked at on a thread of its own as soon as it is
        # shot, while the next is being shot; its frames are let go after.
        work: queue.Queue = queue.Queue()
        failed: list[BaseException] = []

        def looker() -> None:
            while True:
                i = work.get()
                if i is None:
                    return
                try:
                    if not failed:
                        disc.look(i, captured)
                except BaseException as exc:      # reported once the run gets there
                    failed.append(exc)
                for key in [k for k in list(captured)
                            if k == f"r{i}" or k.startswith(f"p{i}.")]:
                    captured.pop(key, None)

        thread = None
        if disc is not None:
            disc.begin(rest, bulb or None)
            thread = threading.Thread(target=looker, name="layoutval-look", daemon=True)
            thread.start()
        keep("rest", rest)
        before = [small(rest[-1])]
        at_rest = before[0]
        moved_since_rest = False
        n = len(steps)
        try:
            for k, step in enumerate(steps):
                if step["id"] == "rest":
                    continue
                if step["kind"] == "rest" and not moved_since_rest:
                    if disc is not None:
                        work.put(step["index"])          # nothing to come back from
                    continue
                self._say("teach", step["name"], k + 1, n)
                since = self._send(step["set"])
                got = watch(self._display_frames(since), before, self.level)
                if step["kind"] == "rest":
                    got = self._come_back(got, at_rest)
                    at_rest = got.last
                keep(step["id"], got.shots)
                before = [small(f) for f in got.shots] if got.flashing else [got.last]
                if step["kind"] == "rest":
                    moved_since_rest = False
                    if disc is not None:
                        work.put(step["index"])
                else:
                    moved_since_rest = moved_since_rest or got.changed
        finally:
            if thread is not None:
                work.put(None)
        self._send({})
        self._say("teach", "learning what every signal showed")
        if thread is not None:
            thread.join()
            if failed:
                raise failed[0]
            lib, summary = disc.learn(captured, bulb=bulb or None)
        else:
            lib, summary = L.learn(s.plan, _ShotFiles(files))
        sweeps = summary.pop("_sweeps", [])
        drift = summary.pop("_drift", None)
        problems = summary.setdefault("problems", [])
        unknown_lamps = []
        if bulb:
            report = L.learn_bulb_check(lib, bulb, problems)
            unknown_lamps = _lamp_like(lib, [d for d in report if d["name"] == "unknown"],
                                       len(bulb))
            _add_unknown_lamps(lib, unknown_lamps, bulb)
        summary["layout"] = L.layout(lib)
        summary["unknown_lamps"] = [d["centre"] for d in unknown_lamps]
        with s.lock:
            s._teach_sweeps = sweeps
            names = s._name_elements()
            dropped = _drop_changing(s.profile, drift) if drift is not None else []
            s.changing_mask = drift if drift is not None and drift.any() else None
            s.lamp_lib = lib
            s.teach_summary = summary
            profile = s.profile
        lib.save(self.folder)
        if profile is not None:
            try:                  # the elements as named, for the record
                profile.save(self.folder / "profile.json")
            except Exception as exc:
                self._log(f"auto: the profile was not saved: {exc}")
        named = sorted({n for n in names.values() if n != "unknown"} - set(dropped))
        unknown = sum(1 for k, n in names.items() if n == "unknown" and k not in dropped)
        summary["elements_named"] = named
        summary["elements_unknown"] = unknown
        summary["not_checked_changes_by_itself"] = dropped
        (self.folder / "teach.json").write_text(json.dumps(summary, indent=2, default=str))
        if s.profile is not None and s.reference is not None:
            cv2.imwrite(str(self.folder / "elements.png"), L.draw_elements(s.reference, s.profile))
        if drift is not None and drift.any() and s.reference is not None:
            shown = s.reference.copy()
            shown[drift] = (0.5 * shown[drift] + [0, 0, 128]).astype(np.uint8)
            cv2.imwrite(str(self.folder / "changed-by-itself.png"), shown)
        if bulb:
            pic = np.max(np.stack(bulb), axis=0)
            cv2.imwrite(str(self.folder / "lamps.png"), L.draw_lamps(pic, lib.find(bulb)))
        self.summary["teach"] = {k: summary.get(k) for k in (
            "signals_tried", "no_effect", "changes_with_time", "layout", "unknown_lamps",
            "elements_named", "elements_unknown", "not_checked_changes_by_itself", "problems")}
        self.summary["teach"]["lamps"] = [lamp.name for lamp in lib.lamps]
        rows = []
        for lamp in lib.lamps:
            if not lamp.states:
                continue
            by = sorted({st.get("signal", "") for st in lamp.states.values()})
            flashes = any(st.get("flashing") for st in lamp.states.values())
            rows.append({"id": lamp.name, "verdict": "PASS",
                         "detail": f"lamp, {L.hue_name(lamp.hue)}" + (", flashes" if flashes else "")
                                   + f" at {lamp.centre[0]:.0f},{lamp.centre[1]:.0f}"
                                   + (f"; lit by {len(by)} signals" if len(by) > 1 else "")})
        rows += [{"id": lamp.name, "verdict": "REVIEW",
                  "detail": f"lit in the bulb check at {lamp.centre[0]:.0f},{lamp.centre[1]:.0f}, "
                            f"{L.hue_name(lamp.hue)}; no signal lit it"}
                 for lamp in lib.lamps if lamp.name.startswith("Unknown lamp")]
        rows += [{"id": nm, "verdict": "PASS", "detail": "element"} for nm in named]
        if unknown:
            rows.append({"id": "unknown elements", "verdict": "PASS",
                         "detail": f"{unknown} on the rest screen no signal changed"})
        if dropped:
            rows.append({"id": "not checked", "verdict": "PASS",
                         "detail": f"{len(dropped)} element(s) that change on their own "
                                   f"(a counter, a clock): {', '.join(dropped[:6])}"})
        rows += [{"id": "teach", "verdict": "REVIEW", "detail": p} for p in problems]
        return lib, rows

    # -- every element on its own -----------------------------------------------

    def _keep_lamp_picture(self, lib, step, shots, found, results) -> None:
        """The lamp a lamp-check step lights, cut out of the shot it is brightest in."""
        from layoutval import lamps as L
        from layoutval.elementsheet import COLOURS, picture

        name = step.get("lamp", step["name"])
        lamp = lib.get(name)
        if lamp is None or not shots:
            return
        d = next((d for d in found if d.name == name), None)
        box = d.box if d is not None else lamp.box
        x, y, w, h = box
        lit = [float(L.lit_delta(lib.baseline, s)[y:y + h, x:x + w].sum()) for s in shots]
        r = next((r for r in results if r.name == name), None)
        verdict = r.verdict if r is not None else "PASS"
        self._lamp_pics.setdefault(name, []).append(self._lamp_picture(
            shots[int(np.argmax(lit))], box, step.get("state") or "on", verdict, lamp,
            r.reason if r is not None and r.reason else ""))

    @staticmethod
    def _lamp_picture(img, box, state: str, verdict: str, lamp, reason: str = "") -> dict:
        from layoutval import lamps as L
        from layoutval.elementsheet import COLOURS, picture

        return {"state": state, "verdict": verdict, "reason": reason,
                "colour": L.hue_name(lamp.hue),
                "img": picture(img, box, COLOURS.get(verdict, (200, 200, 200)))}

    def _element_pictures(self, val, rows: list[dict[str, Any]] = ()) -> str | None:
        """Every element and lamp cut out against its name: elements.html, elements/.

        Headed by what did not pass among ``rows``, with the bulb check's and
        the lamp check's pictures when the run took them.
        """
        from layoutval import elementsheet as E

        s = self.s
        items: list[dict[str, Any]] = []
        for name, pics in self._lamp_pics.items():
            verdict = _worst([p["verdict"] for p in pics])
            reasons = sorted({p["reason"] for p in pics if p["reason"]})
            items.append({"name": name, "kind": "lamp", "verdict": verdict,
                          "detail": ", ".join([pics[0]["colour"]] + reasons),
                          "pictures": [(p["state"], p["img"]) for p in pics[:3]]})
        now = None
        shot = s.out_dir / f"{self.stamp}-validate-rectified.png"
        if shot.is_file():
            now = cv2.imread(str(shot))
        results = {r["element_id"]: r for r in
                   (((val.report or {}).get("elements") or []) if val is not None else [])}
        if s.profile is not None and s.reference is not None:
            for e in s.profile.elements:
                r = results.get(e.id, {})
                m = r.get("measurement") or {}
                verdict = r.get("verdict", "")
                colour = E.COLOURS.get(verdict, (200, 200, 200))
                pics = [("reference", E.picture(s.reference, e.bbox))]
                if now is not None and r:
                    pics.append(("this run", E.picture(now, _shift(e.bbox, m.get("dx"), m.get("dy")),
                                                       colour)))
                if m.get("dx") is not None and m.get("dy") is not None:
                    detail = f"moved {m['dx']:+.2f}, {m['dy']:+.2f} px"
                else:
                    detail = r.get("reason") or ("not measured" if r else "")
                if r.get("reason") and m.get("dx") is not None:
                    detail += f" ({r['reason']})"
                items.append({"name": e.id, "kind": "element", "verdict": verdict,
                              "detail": detail, "pictures": pics})
        if not items:
            return None
        problems = [r for r in rows if r.get("verdict") not in ("PASS", "OK", "TAUGHT")]
        views = []
        for name, label in (("bulb-check.png", "the bulb check, as taken from the start-up"),
                            ("lamp-check.png", "the lamp check: all lamps, or a state that did not pass")):
            path = self.folder / name
            img = cv2.imread(str(path)) if path.is_file() else None
            if img is not None:
                views.append((label, img))
        E.write(self.folder, items, f"Elements, {self.folder.name}", problems, views)
        return f"{self.folder.name}/elements.html"

    def _look_again(self, seconds: float = 2.5, keep: int = 12) -> list[np.ndarray]:
        """The state as it is, shot again over a longer window."""
        frames = []
        end = time.monotonic() + seconds
        for t, _, d in self._display_frames(time.monotonic()):
            frames.append(d)
            if t >= end:
                break
        if len(frames) > keep:
            idx = np.linspace(0, len(frames) - 1, keep).round().astype(int)
            frames = [frames[k] for k in sorted(set(idx.tolist()))]
        return frames

    def _come_back(self, got: Settled, at_rest: np.ndarray) -> Settled:
        """Back at rest: wait while the screen is still on its way back to how it was.

        What has not come back after :data:`RETURN_S` changed for good (a trip
        counter, a clock) and is left for the teach-in to find as such.
        """
        end = time.monotonic() + RETURN_S
        while (changed_px(got.last, at_rest, self.level) >= CHANGE_PX
               and time.monotonic() < end):
            again = watch(self._display_frames(time.monotonic()), [got.last], self.level,
                          react_s=1.0)
            if not again.changed:
                break
            got = again
        return got

    def _elements_of(self, step: dict[str, Any], lib) -> np.ndarray | None:
        """Where the step's signals change elements (a gauge, a readout), and
        where the screen changes by itself: a change there is not a lamp."""
        s = self.s
        disc = getattr(s, "discovery", None)
        shape = lib.baseline.shape[:2]
        mask = (s.changing_mask.copy() if s.changing_mask is not None
                and s.changing_mask.shape == shape else np.zeros(shape, bool))
        if disc is None or not s._teach_sweeps:
            return mask
        signals = set(step["set"])
        names = {sp.name for sp in disc.signals if sp.signal in signals}
        boxes = [b for sw in s._teach_sweeps if sw["name"] in names for b in sw["regions"]]
        if boxes:
            mask |= _region_mask(shape, boxes)
        return mask

    def _bulb_test(self, lib, bulb: list[np.ndarray],
                   rest: list[np.ndarray]) -> list[dict[str, Any]]:
        """A later run's bulb check: every lamp lit, each in its place, nothing else."""
        from layoutval import lamps as L

        if not bulb:
            return self._no_bulb_check(0)
        # What lit is measured against the rest screen of this power cycle --
        # this camera, this mapping, this minute -- as the lamp check does,
        # not against the teach-in's.
        lib.baseline = np.median(np.stack(rest), axis=0).astype(np.uint8)

        # What sweeps through the bulb check (a needle, its readout) is no
        # lamp; and a lamp that is not part of it (a shift light the sweeping
        # needle trips) is not judged by it.
        in_check = {lamp.name for lamp in lib.lamps if lamp.slot_check is not None}
        found = [d for d in lib.find(bulb)
                 if d.name in in_check
                 or d.name == "unknown" and _lamp_like(lib, [d.to_dict()], len(bulb))]
        # One not read among the rest is looked for in its place, by its own
        # symbol (the lamps beside it lit can hide it from find).
        named = {d.name for d in found}
        for lamp in lib.lamps:
            if lamp.slot_check is not None and lamp.name not in named:
                d = lib.at_place(lamp, bulb, lamp.slot_check)
                if d is not None:
                    found = [u for u in found
                             if not (u.name == "unknown" and L._overlap(u.box, d.box) > 0.3)]
                    found.append(d)
        expected = [lamp.name for lamp in lib.lamps if lamp.slot_check is not None]
        results = L.check(lib, found, expected, where="check")
        v = L.verdict(results)
        self.summary["bulb_check_test"] = v
        lit = sum(r.found for r in results if r.expected)
        if expected and lit == 0:
            # Not one lamp: not 31 faults but one -- no bulb check seen.
            return self._no_bulb_check(len(bulb))
        rows = [{"id": "Bulb check", "verdict": v,
                 "detail": f"{lit} of {len(expected)} lamps lit at power-up"}]
        rows += [{"id": r.name, "verdict": r.verdict, "detail": f"{r.reason} (bulb check)"}
                 for r in results if r.verdict != "PASS"]
        return rows

    def _no_bulb_check(self, picked: int) -> list[dict[str, Any]]:
        seen = self.summary.get("bulb_check", {})
        self.summary["bulb_check_test"] = "FAIL"
        return [{"id": "Bulb check", "verdict": "FAIL",
                 "detail": f"no lamp seen lit at power-up ({picked} start-up frame(s) taken for "
                           f"the bulb check, of {seen.get('startup_frames', '?')} recorded over "
                           f"{seen.get('startup_s', '?')} s): the cluster did not run its bulb "
                           "check, or the camera did not catch it"}]

    # -- 5. lamp check ----------------------------------------------------------

    def _check(self, lib) -> list[dict[str, Any]]:
        from layoutval import lamps as L

        s = self.s
        disc = getattr(s, "discovery", None)
        steps = disc.check_steps(lib) if disc is not None else s.plan.check_steps()
        tol = s.profile.defaults if s.profile is not None else None
        before = None
        rows, verdicts, report = [], [], []
        picture = None
        all_state = disc.all_state(lib) if disc is not None else {}
        n = len(steps)
        for i, step in enumerate(steps):
            label = step["name"] + (f" ({step['state']})" if step.get("state") else "")
            self._say("lamps", f"checking {label}", i + 1, n)
            since = self._send(step["set"])
            got = watch(self._display_frames(since), before or [], self.level,
                        react_s=REACT_S if before else 0.3)
            before = [small(f) for f in got.shots] if got.flashing else [got.last]
            if step["id"] == "rest0":
                # The rest screen as it is now: what is lit is measured against
                # it, not against the teach-in's, minutes or days ago.
                lib.baseline = np.median(np.stack(got.shots), axis=0).astype(np.uint8)
                continue
            if step["kind"] == "release":
                continue
            found = lib.find(got.shots)
            flashing = None
            if step["kind"] == "lamp":
                expected, where = [step.get("lamp", step["name"])], "alone"
                if "flashing" in step:
                    flashing = {expected[0]: bool(step["flashing"])}
            elif step["kind"] == "all":
                if disc is not None:
                    flashing = _lit_by(lib, all_state)
                    expected = list(flashing)
                else:
                    expected = [lamp.name for lamp in lib.lamps if lamp.slot_all is not None]
                where = "all"
            else:
                expected, where = [], ""
            aside = self._elements_of(step, lib)

            def judge(found):
                results = L.check(lib, found, expected, where=where,
                                  tol_warn=tol.tol_warn if tol else 1.5,
                                  tol_fail=tol.tol_fail if tol else 2.5, flashing=flashing)
                return [r for r in results if not (
                    r.name == "unknown" and r.centre is not None and aside is not None
                    and aside[min(aside.shape[0] - 1, int(r.centre[1])),
                              min(aside.shape[1] - 1, int(r.centre[0]))])]

            results = judge(found)
            second = False
            if L.verdict(results) != "PASS":
                # A second, longer look at the same state before saying so: a
                # few frames can all land in one half of a flash.
                self._say("lamps", f"checking {label} again", i + 1, n)
                again = self._look_again()
                found = lib.find(again)
                results, second = judge(found), True
                got = Settled(shots=again, last=small(again[-1]), changed=False,
                              flashing=False, settled=True, seconds=0.0)
            v = L.verdict(results)
            verdicts.append(v)
            report.append({"step": step["name"], "state": step.get("state"), "verdict": v,
                           "lamps": [r.to_dict() for r in results]})
            if step["kind"] == "lamp":
                self._keep_lamp_picture(lib, step, got.shots, found, results)
            if step["kind"] == "all" or (v != "PASS" and picture is None):
                picture = L.draw_lamps(np.max(np.stack(got.shots), axis=0), found, results)
            for r in results:
                if r.verdict == "PASS":
                    continue
                own = step["kind"] == "lamp" and r.name == step.get("lamp")
                off = f"{math.hypot(r.dx, r.dy):.2f} px " if r.dx is not None else ""
                rows.append({"id": r.name, "verdict": r.verdict,
                             "detail": f"{off}{r.reason}" + ("" if own else f" (with {label})")
                                       + (", seen twice" if second else "")})
        self._send({})
        verdict = _worst(verdicts) if verdicts else "PASS"
        passed = sum(1 for v in verdicts if v == "PASS")
        self.summary["lamp_check"] = f"{verdict}, {passed} of {len(verdicts)} states as expected"
        (self.folder / "lamp-check.json").write_text(json.dumps(
            {"verdict": verdict, "steps": report}, indent=2, default=str))
        if picture is not None:
            cv2.imwrite(str(self.folder / "lamp-check.png"), picture)
        return [{"id": "Lamp check", "verdict": verdict,
                 "detail": f"{passed} of {len(verdicts)} lamp states as expected"}] + rows


def _shift(box, dx, dy) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x + (dx or 0.0), y + (dy or 0.0), w, h


def _lamp_like(lib, dets: list[dict[str, Any]], shots: int) -> list[dict[str, Any]]:
    """The unknown symbols of a bulb check that are lamps: lit throughout, still,
    sized like the lamps, and not on one -- a needle swept through it is none."""
    w0, h0 = lib.lamp_size
    out = []
    for d in dets:
        x, y, w, h = d["box"]
        if d.get("lit_in", shots) < shots or d.get("moved_px", 0.0) > 2.0:
            continue
        if not (0.6 <= w / w0 <= 1.7 and 0.6 <= h / h0 <= 1.7):
            continue
        if any(abs(d["centre"][0] - lamp.centre[0]) < w0 and abs(d["centre"][1] - lamp.centre[1]) < h0
               for lamp in lib.lamps):
            continue
        out.append(d)
    return out


def _region_mask(shape, boxes, pad: int = 6) -> np.ndarray:
    m = np.zeros(shape, bool)
    for x, y, w, h in boxes:
        m[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = True
    return m


def _add_unknown_lamps(lib, dets: list[dict[str, Any]], bulb: list[np.ndarray]) -> None:
    """A lamp only the bulb check lights joins the lamps, by its symbol and place,
    as "Unknown lamp 1", ...: a later bulb check expects it, and lit at any
    other time it is named, not "unknown"."""
    from layoutval import lamps as L

    if not dets:
        return
    level = L._threshold(lib.lit_floor, L.LIT_MIN)
    deltas = [L.lit_delta(lib.baseline, f) for f in bulb]
    blobs = L.blobs(np.median(np.stack(deltas), axis=0), level)
    for k, d in enumerate(sorted(dets, key=lambda d: (d["centre"][1], d["centre"][0])), 1):
        near = [b for b in blobs if L._overlap(b.box, tuple(d["box"])) > 0.3]
        if not near:
            continue
        c = L._candidate(max(near, key=lambda b: b.area), deltas, bulb, level)
        lamp = L.Lamp(name=f"Unknown lamp {k}", set={}, blinks=False, box=c["box"],
                      centre=c["centre"], template=c["template"], hue=c["hue"],
                      colour_ratio=c["ratio"], strength=c["strength"], offset=c["offset"],
                      states={}, slot_check=c["centre"], fixed=True)
        lib.lamps.append(lamp)
    L._set_zone(lib)


def _drop_changing(profile, drift: np.ndarray, share: float = 0.2) -> list[str]:
    """Take out of the profile the elements that changed on their own while it was
    taught -- an odometer, a trip counter, a clock -- and say which.

    Compared later they would fail for being right: the reading moves on.
    """
    if profile is None or drift is None or not drift.any():
        return []
    gone = []
    for spec in list(profile):
        x, y, w, h = (int(round(v)) for v in spec.bbox)
        box = drift[max(0, y):y + h, max(0, x):x + w]
        if box.size and float(box.mean()) >= share:
            gone.append(spec.id)
    profile.elements = [spec for spec in profile if spec.id not in gone]
    return gone


def _lit_by(lib, state: dict[str, Any]) -> dict[str, bool]:
    """The lamps ``state`` lights, and whether each flashes in the state it is in."""
    out = {}
    for lamp in lib.lamps:
        if not lamp.states:
            continue                       # nothing on the bus lights it
        for st in lamp.states.values():
            if all(state.get(k) == v for k, v in st["set"].items()):
                out[lamp.name] = bool(st.get("flashing"))
                break
    return out


def _worst(verdicts: list[str]) -> str:
    rank = {"FAIL": 0, "FAILED": 0, "REVIEW": 1, "PASS": 2, "OK": 2}
    if not verdicts:
        return "PASS"
    w = min(verdicts, key=lambda v: rank.get(v, 1))
    return {"FAILED": "FAIL", "OK": "PASS"}.get(w, w)

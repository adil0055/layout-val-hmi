"""A synthetic cluster and a virtual camera.

This exists so the pipeline can be developed, tested and demonstrated without a
bench, and so that defects can be *injected* with a known ground truth -- which
is the only way to know whether a measurement chain reports the right number.

The virtual camera deliberately reproduces the artefacts specific to
photographing a display, because they are what the real rig has to survive:

* **perspective and lens distortion**, so rectification is actually exercised;
* **backlight PWM banding**, whose phase advances per frame -- median-stacking
  several frames is what removes it;
* **defocus**, the recommended anti-moiré measure (a slightly soft image with no
  moiré measures better than a sharp one with it);
* **thermal drift**, a slow sub-pixel creep across a long run;
* **shot noise** and 8-bit quantisation.

It is a test fixture, not a model of any particular camera.  Numbers measured
against it say the code is correct; they say nothing about what a real rig can
resolve.  That is what the repeatability study is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# --------------------------------------------------------------------------
# the synthetic cluster
# --------------------------------------------------------------------------

BG = (16, 12, 10)
AMBER = (40, 150, 225)
RED = (60, 60, 220)
GREEN = (120, 200, 110)
WHITE = (225, 228, 230)
PLATE = (110, 105, 100)


@dataclass
class ClusterDisplay:
    """Renders a framebuffer.  Coordinates here *are* display coordinates."""

    size: tuple[int, int] = (960, 360)
    theme: str = "day"
    supersample: int = 4
    """Elements are drawn on a canvas this many times larger and boxed down, so
    fractional positions survive rendering."""

    state: dict[str, Any] = field(default_factory=dict)

    # Injected defects, in display pixels.  Ground truth for the tests.
    offsets: dict[str, tuple[float, float]] = field(default_factory=dict)
    swapped: set[str] = field(default_factory=set)
    hidden: set[str] = field(default_factory=set)
    artefacts: list[tuple[int, int, int, int]] = field(default_factory=list)
    needle_pivot_offset: tuple[float, float] = (0.0, 0.0)
    needle_angle_offset_deg: float = 0.0

    # Nominal geometry, mirrored by examples/design_export.json.
    LAYOUT: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: {
            "TELLTALE_BATTERY_LOW": (120, 74, 60, 40),
            "TELLTALE_OIL_PRESSURE": (216, 74, 52, 40),
            "TELLTALE_ABS": (304, 74, 54, 40),
            "LABEL_SPEED_UNITS": (432, 250, 62, 22),
            "GAUGE_PLATE": (660, 100, 200, 200),
            "FUEL_BAR": (120, 300, 120, 18),
            "NEEDLE_SPEED": (676, 116, 168, 168),
        }
    )

    GAUGE_PIVOT = (760.0, 200.0)
    FUEL_TRAVEL = (300.0, 0.0)

    def __post_init__(self) -> None:
        self.state.setdefault("TELLTALE_BATTERY_LOW", False)
        self.state.setdefault("TELLTALE_OIL_PRESSURE", False)
        self.state.setdefault("TELLTALE_ABS", False)
        self.state.setdefault("FUEL_LEVEL", 0.0)
        self.state.setdefault("SPEED", 0.0)

    # -- helpers ------------------------------------------------------------

    def _o(self, eid: str) -> tuple[float, float]:
        return self.offsets.get(eid, (0.0, 0.0))

    def _box(self, eid: str) -> tuple[float, float, float, float]:
        x, y, w, h = self.LAYOUT[eid]
        dx, dy = self._o(eid)
        return x + dx, y + dy, w, h

    def _dim(self, colour: tuple[int, int, int]) -> tuple[int, int, int]:
        # Night theme is a gamma change as well as a gain change, which is
        # exactly why each theme is taught separately rather than derived.
        if self.theme != "night":
            return colour
        return tuple(int(round(255.0 * (c / 255.0) ** 1.9 * 0.55)) for c in colour)

    # -- element drawing ----------------------------------------------------

    # Drawing happens on a supersampled canvas and is boxed down afterwards, so
    # an element can be placed at a genuine fractional display coordinate.  A
    # simulator that quantised every position to a whole pixel could not be used
    # to check a sub-pixel measurement against ground truth.
    def _p(self, x: float, y: float) -> tuple[int, int]:
        """Display coordinate -> supersampled pixel index.

        Box-filtering by S maps output pixel ``i`` to the mean of input
        subpixels ``[i*S, (i+1)*S)``, so subpixel ``j`` sits at display
        coordinate ``(j + 0.5)/S - 0.5``.  Inverting that is where the
        ``(S-1)/2`` comes from; dropping it puts a fixed
        ``0.5 - 0.5/S`` pixel offset into every element, which then looks like a
        real systematic bias in the measurements.
        """
        S = self.supersample
        off = (S - 1) / 2.0
        return int(round(x * S + off)), int(round(y * S + off))

    def _t(self, thickness: int) -> int:
        return max(1, int(round(thickness * self.supersample)))

    def _battery(self, img: np.ndarray) -> None:
        eid = "TELLTALE_BATTERY_LOW"
        if not self.state[eid] or eid in self.hidden:
            return
        x, y, w, h = self._box(eid)
        c = self._dim(RED if eid not in self.swapped else GREEN)
        t = self._t(2)
        cv2.rectangle(img, self._p(x + 4, y + 8), self._p(x + w - 4, y + h - 4), c, t, cv2.LINE_AA)
        cv2.rectangle(img, self._p(x + 12, y + 4), self._p(x + 20, y + 8), c, -1, cv2.LINE_AA)
        cv2.rectangle(img, self._p(x + w - 20, y + 4), self._p(x + w - 12, y + 8), c, -1, cv2.LINE_AA)
        if eid in self.swapped:
            # A different glyph in the right place: correct position, wrong
            # content.  ZNCC catches it; a position check alone does not.
            cv2.line(img, self._p(x + 10, y + h - 10), self._p(x + w - 10, y + 12), c, t, cv2.LINE_AA)
        else:
            cy = y + h / 2 + 2
            cv2.line(img, self._p(x + 12, cy), self._p(x + 22, cy), c, t, cv2.LINE_AA)
            cv2.line(img, self._p(x + 17, cy - 5), self._p(x + 17, cy + 5), c, t, cv2.LINE_AA)
            cv2.line(img, self._p(x + w - 22, cy), self._p(x + w - 12, cy), c, t, cv2.LINE_AA)

    def _oil(self, img: np.ndarray) -> None:
        eid = "TELLTALE_OIL_PRESSURE"
        if not self.state[eid] or eid in self.hidden:
            return
        x, y, w, h = self._box(eid)
        c = self._dim(RED)
        S = self.supersample
        cv2.ellipse(img, self._p(x + w / 2, y + h / 2 + 2),
                    (int(w / 3 * S), int(h / 4 * S)), 0, 0, 360, c, self._t(2), cv2.LINE_AA)
        cv2.line(img, self._p(x + 6, y + h / 2 + 2), self._p(x + w / 3, y + h / 2 + 2), c, self._t(2), cv2.LINE_AA)
        cv2.circle(img, self._p(x + w / 2, y + 10), int(4 * S), c, self._t(2), cv2.LINE_AA)

    def _abs(self, img: np.ndarray) -> None:
        eid = "TELLTALE_ABS"
        if not self.state[eid] or eid in self.hidden:
            return
        x, y, w, h = self._box(eid)
        c = self._dim(AMBER)
        S = self.supersample
        cv2.circle(img, self._p(x + w / 2, y + h / 2), int((min(w, h) / 2 - 2) * S), c, self._t(2), cv2.LINE_AA)
        cv2.putText(img, "ABS", self._p(x + 10, y + h / 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36 * S, c, self._t(1), cv2.LINE_AA)

    def _speed_units(self, img: np.ndarray) -> None:
        eid = "LABEL_SPEED_UNITS"
        if eid in self.hidden:
            return
        x, y, w, h = self._box(eid)
        cv2.putText(img, "km/h", self._p(x, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * self.supersample, self._dim(WHITE), self._t(1), cv2.LINE_AA)

    def _gauge_plate(self, img: np.ndarray) -> None:
        eid = "GAUGE_PLATE"
        if eid in self.hidden:
            return
        dx, dy = self._o(eid)
        cx, cy = self.GAUGE_PIVOT[0] + dx, self.GAUGE_PIVOT[1] + dy
        c = self._dim(PLATE)
        S = self.supersample
        cv2.circle(img, self._p(cx, cy), int(96 * S), c, self._t(2), cv2.LINE_AA)
        for i in range(13):
            a = math.radians(210 - i * 20)
            r0, r1 = (78, 94) if i % 2 == 0 else (86, 94)
            cv2.line(
                img,
                self._p(cx + r0 * math.cos(a), cy - r0 * math.sin(a)),
                self._p(cx + r1 * math.cos(a), cy - r1 * math.sin(a)),
                c,
                self._t(2 if i % 2 == 0 else 1),
                cv2.LINE_AA,
            )

    def _fuel_bar(self, img: np.ndarray) -> None:
        eid = "FUEL_BAR"
        if eid in self.hidden:
            return
        x, y, w, h = self.LAYOUT[eid]
        dx, dy = self._o(eid)
        t = float(np.clip(self.state["FUEL_LEVEL"], 0.0, 1.0))
        x = x + dx + self.FUEL_TRAVEL[0] * t
        y = y + dy + self.FUEL_TRAVEL[1] * t
        c = self._dim(GREEN)
        cv2.rectangle(img, self._p(x, y), self._p(x + w, y + h), c, -1, cv2.LINE_AA)
        cv2.rectangle(img, self._p(x + 4, y + 4), self._p(x + w - 4, y + h - 4),
                      self._dim(BG), self._t(1), cv2.LINE_AA)

    def needle_angle_deg(self) -> float:
        """Nominal needle angle for the current speed, CCW-positive on screen."""
        t = float(np.clip(self.state["SPEED"] / 240.0, 0.0, 1.0))
        return 210.0 - 240.0 * t + self.needle_angle_offset_deg

    def needle_pivot(self) -> tuple[float, float]:
        dx, dy = self._o("NEEDLE_SPEED")
        return (
            self.GAUGE_PIVOT[0] + dx + self.needle_pivot_offset[0],
            self.GAUGE_PIVOT[1] + dy + self.needle_pivot_offset[1],
        )

    def _needle(self, img: np.ndarray) -> None:
        if "NEEDLE_SPEED" in self.hidden:
            return
        cx, cy = self.needle_pivot()
        a = math.radians(self.needle_angle_deg())
        c = self._dim(RED)
        cv2.line(img, self._p(cx, cy), self._p(cx + 76 * math.cos(a), cy - 76 * math.sin(a)),
                 c, self._t(4), cv2.LINE_AA)
        # A hub, as every real needle has.  It is what makes naive
        # extreme-point pivot estimation biased, so the simulator keeps it.
        cv2.circle(img, self._p(cx, cy), int(7 * self.supersample), c, -1, cv2.LINE_AA)

    # -- render -------------------------------------------------------------

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        w, h = self.size
        if self.supersample == 1:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def render(self) -> np.ndarray:
        w, h = self.size
        S = self.supersample
        img = np.full((h * S, w * S, 3), self._dim(BG), np.uint8)
        self._gauge_plate(img)
        self._battery(img)
        self._oil(img)
        self._abs(img)
        self._speed_units(img)
        self._fuel_bar(img)
        self._needle(img)
        for x, y, aw, ah in self.artefacts:
            cv2.rectangle(img, self._p(x, y), self._p(x + aw, y + ah), self._dim(AMBER), -1)
        return self._downsample(img)

    def render_checkerboard(
        self, pattern_size: tuple[int, int] = (9, 6), square_px: int = 40, origin: tuple[int, int] = (60, 20)
    ) -> np.ndarray:
        """The layout-calibration screen a test build should provide.

        A one-off diagnostic screen like this is the highest-leverage thing to
        ask the HMI team for: it gives a direct, exact correspondence between
        framebuffer coordinates and camera coordinates, and it pays for itself
        permanently.
        """
        w, h = self.size
        S = self.supersample
        cols, rows = pattern_size[0] + 1, pattern_size[1] + 1
        # Built from the coordinate grid rather than with rectangle() so the
        # square boundaries land on exact display coordinates.  That makes inner
        # corner (c, r) sit at origin + ((c+1)*square, (r+1)*square) with no
        # half-pixel convention to remember -- which matters, because this frame
        # is what every downstream number is referred to.
        xs = (np.arange(w * S) + 0.5) / S - 0.5
        ys = (np.arange(h * S) + 0.5) / S - 0.5
        cx = np.floor((xs - origin[0]) / square_px).astype(np.int64)
        cy = np.floor((ys - origin[1]) / square_px).astype(np.int64)
        inside_x = (cx >= 0) & (cx < cols)
        inside_y = (cy >= 0) & (cy < rows)
        board = ((cx[None, :] + cy[:, None]) % 2 == 0) & inside_x[None, :] & inside_y[:, None]
        img = np.zeros((h * S, w * S, 3), np.uint8)
        img[board] = 255
        return self._downsample(img)

    def render_white(self) -> np.ndarray:
        w, h = self.size
        return np.full((h, w, 3), 245, np.uint8)


# --------------------------------------------------------------------------
# the virtual camera
# --------------------------------------------------------------------------


def _distortion_map(K: np.ndarray, dist: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map taking each *distorted* output pixel to its source in the ideal image.

    This is the opposite direction from :func:`cv2.initUndistortRectifyMap`,
    which is what an undistorter needs; synthesising a distorted image needs the
    inverse, obtained here with :func:`cv2.undistortPoints`.
    """
    w, h = size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    ideal = und @ K[:2, :2].T + K[:2, 2]
    return (
        ideal[:, 0].reshape(h, w).astype(np.float32),
        ideal[:, 1].reshape(h, w).astype(np.float32),
    )


@dataclass
class Reflection:
    """Something bright in the room, reflected in the cluster's cover glass.

    Laid over the camera frame rather than drawn on the display, because that
    is where it lives: it belongs to the room and the glass, so when the camera
    moves it moves across the display's content instead of with it. Positions
    and sizes are fractions of the sensor.
    """

    centre: tuple[float, float] = (0.5, 0.5)
    size: tuple[float, float] = (0.2, 0.2)
    strength: float = 60.0
    """Grey levels it lifts a black pixel to at its centre. It adds in linear
    light, so it lifts bright content far less. Strong enough and pixels clip --
    and a clipped pixel carries no information about what is under it."""

    softness: float = 0.0
    """Blur of the reflection's edges, in pixels. 0 gives a soft Gaussian blob
    -- an out-of-focus window; a few pixels gives a lamp or a window frame,
    with edges sharp enough to look like content."""

    def image(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        cx, cy = self.centre[0] * w, self.centre[1] * h
        sx, sy = self.size[0] * w / 2, self.size[1] * h / 2
        if self.softness <= 0:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            return self.strength * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        out = np.zeros((h, w), np.float32)
        cv2.rectangle(out, (int(cx - sx), int(cy - sy)), (int(cx + sx), int(cy + sy)),
                      float(self.strength), -1)
        return cv2.GaussianBlur(out, (0, 0), self.softness)


@dataclass
class VirtualCamera:
    """Turns a framebuffer into something that looks like a photograph of one."""

    display_size: tuple[int, int] = (960, 360)
    sensor_size: tuple[int, int] = (1600, 900)
    sampling_ratio: float = 1.6
    """Camera pixels per display pixel.  Set it below 2 to see the pipeline's
    accuracy degrade the way a real under-sampled rig does."""

    tilt_deg: float = 2.0
    roll_deg: float = 0.7
    k1: float = -0.09
    k2: float = 0.02
    defocus_sigma: float = 0.9
    """Roughly one display pixel of defocus -- the recommended anti-moiré
    measure when the sampling ratio cannot be moved off a near-integer value."""

    noise_sigma: float = 1.2
    pwm_amplitude: float = 0.10
    pwm_period_px: float = 90.0
    drift_per_frame_px: float = 0.0
    """Slow thermal creep, applied cumulatively.  Set it non-zero to exercise the
    drift tracker and its alarm."""

    seed: int = 0

    glare: list[Reflection] = field(default_factory=list)
    """Reflections in the cover glass. Empty by default."""

    def __post_init__(self) -> None:
        dw, dh = self.display_size
        sw, sh = self.sensor_size
        s = self.sampling_ratio
        # Centre the display in the sensor, then tilt and roll it.
        cx, cy = sw / 2.0, sh / 2.0
        corners_d = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], np.float64)
        centred = (corners_d - [dw / 2.0, dh / 2.0]) * s

        tilt = math.radians(self.tilt_deg)
        roll = math.radians(self.roll_deg)
        z = 1500.0
        pts3 = np.column_stack([centred, np.zeros(4)])
        Rx = np.array(
            [[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]
        )
        Rz = np.array(
            [[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]]
        )
        pts3 = pts3 @ (Rz @ Rx).T
        pts3[:, 2] += z
        f = z
        corners_c = np.column_stack(
            [cx + f * pts3[:, 0] / pts3[:, 2], cy + f * pts3[:, 1] / pts3[:, 2]]
        )
        self.H_true = cv2.getPerspectiveTransform(
            corners_d.astype(np.float32), corners_c.astype(np.float32)
        ).astype(np.float64)

        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
        self.dist = np.array([self.k1, self.k2, 0.0, 0.0, 0.0], np.float64)
        self._map = _distortion_map(self.K, self.dist, self.sensor_size)
        self._rng = np.random.default_rng(self.seed)
        self._frame = 0

    def shoot(self, display_img: np.ndarray) -> np.ndarray:
        """One frame, with all of the artefacts applied."""
        sw, sh = self.sensor_size
        H = self.H_true.copy()
        if self.drift_per_frame_px:
            d = self.drift_per_frame_px * self._frame
            H = np.array([[1, 0, d], [0, 1, d * 0.4], [0, 0, 1]], np.float64) @ H

        ideal = cv2.warpPerspective(
            display_img, H, (sw, sh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
        )
        img = cv2.remap(ideal, self._map[0], self._map[1], cv2.INTER_CUBIC)
        img = img.astype(np.float32)

        if self.defocus_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.defocus_sigma)

        if self.pwm_amplitude > 0:
            # Band position advances every frame because the exposure is not an
            # exact multiple of the PWM period -- which is the whole problem.
            phase = (self._frame * 0.37) % 1.0
            rows = np.arange(sh, dtype=np.float32)
            band = 1.0 + self.pwm_amplitude * np.sin(
                2 * np.pi * (rows / self.pwm_period_px + phase)
            )
            img *= band[:, None, None]

        if self.glare:
            # Added in linear light, because that is where light adds: a lamp's
            # reflection lifts a black pixel a long way and a white one hardly
            # at all, which squeezes the content's contrast under it rather than
            # just offsetting it. Adding in encoded values would make glare a
            # pure offset -- the one kind every correlator already ignores.
            # After the flicker banding: that is the backlight's PWM, and the
            # room's light does not share it.
            linear = (np.clip(img, 0, 255) / 255.0) ** 2.2
            for reflection in self.glare:
                linear += (reflection.image((sh, sw)) / 255.0)[..., None] ** 2.2
            img = 255.0 * np.clip(linear, 0, None) ** (1 / 2.2)

        if self.noise_sigma > 0:
            img += self._rng.normal(0.0, self.noise_sigma, img.shape).astype(np.float32)

        self._frame += 1
        return np.clip(img, 0, 255).astype(np.uint8)


class SimulatedRig:
    """A :class:`FrameSource` and a :class:`SignalDriver` in one object.

    Lets the whole pipeline -- teach-in included -- run end to end in a test, with
    ground truth available for every injected defect.
    """

    def __init__(
        self,
        display: ClusterDisplay | None = None,
        camera: VirtualCamera | None = None,
    ) -> None:
        self.display = display or ClusterDisplay()
        self.camera = camera or VirtualCamera(display_size=self.display.size)
        self._screen: str = "main"

    # FrameSource
    def read(self) -> np.ndarray:
        if self._screen == "checkerboard":
            frame = self.display.render_checkerboard()
        elif self._screen == "white":
            frame = self.display.render_white()
        else:
            frame = self.display.render()
        return self.camera.shoot(frame)

    # SignalDriver
    def set(self, signal: str, value: Any) -> None:
        self.display.state[signal] = value

    def show(self, screen: str) -> None:
        """Switch between the normal screen and the diagnostic ones."""
        self._screen = screen


# --------------------------------------------------------------------------
# a cluster in its surround, with markers on the bezel
# --------------------------------------------------------------------------


@dataclass
class BezelPanel:
    """The display set into a bezel that carries a ChArUco board.

    The point of the arrangement is that the markers are part of the *rig*, not
    of the software under test: they are stuck on once, they survive a power
    cycle and a software load, and the cluster never has to draw anything for
    the camera's benefit.  That is the only calibration route that works on a
    production cluster, so it needs to be testable here.

    Everything is in panel pixels.  The display occupies ``display_rect``, and
    the board sits wherever ``board_origin`` puts it.
    """

    panel_size: tuple[int, int] = (2400, 1460)
    #: The opening's proportions match the framebuffer's, because a real module
    #: cannot stretch its own pixels. Getting this wrong here makes any aperture
    #: search look broken when it is the fixture that is impossible.
    display_rect: tuple[int, int, int, int] = (305, 165, 1790, 671)
    squares: tuple[int, int] = (16, 3)
    square_px: int = 99
    #: Board units, as the board object is told them.  Only the ratio to
    #: `square_px` matters: it is what the learned anchor absorbs.
    square_length: float = 30.0
    marker_ratio: float = 0.72
    board_origin: tuple[int, int] = (405, 1090)
    bezel_grey: int = 38

    def __post_init__(self) -> None:
        import cv2 as _cv2

        self.dictionary = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_4X4_100)
        self.board = _cv2.aruco.CharucoBoard(
            self.squares,
            self.square_length,
            self.square_length * self.marker_ratio,
            self.dictionary,
        )

    @property
    def board_size_px(self) -> tuple[int, int]:
        return (self.squares[0] * self.square_px, self.squares[1] * self.square_px)

    def display_from_board(self) -> np.ndarray:
        """The true board-to-display transform, for a test to score against.

        Board units scale to panel pixels by ``square_px / square_length``, then
        the display's own origin is subtracted.
        """
        scale = self.square_px / self.square_length
        ox, oy = self.board_origin
        dx, dy = self.display_rect[0], self.display_rect[1]
        return np.array([
            [scale, 0.0, ox - dx],
            [0.0, scale, oy - dy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def render(self, display_img: np.ndarray) -> np.ndarray:
        """The whole panel: bezel, markers, and the display set into it."""
        w, h = self.panel_size
        panel = np.full((h, w, 3), self.bezel_grey, np.uint8)

        bw, bh = self.board_size_px
        board_img = self.board.generateImage((bw, bh))
        ox, oy = self.board_origin
        panel[oy : oy + bh, ox : ox + bw] = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

        dx, dy, dw, dh = self.display_rect
        fitted = display_img
        if (display_img.shape[1], display_img.shape[0]) != (dw, dh):
            fitted = cv2.resize(display_img, (dw, dh), interpolation=cv2.INTER_AREA)
        panel[dy : dy + dh, dx : dx + dw] = fitted
        return panel


class BezelRig:
    """A :class:`SimulatedRig` whose camera sees the bezel as well as the screen."""

    def __init__(
        self,
        display: ClusterDisplay | None = None,
        panel: BezelPanel | None = None,
        camera: VirtualCamera | None = None,
    ) -> None:
        self.display = display or ClusterDisplay()
        self.panel = panel or BezelPanel(
            display_rect=(305, 165, self.display.size[0], self.display.size[1])
        )
        self.camera = camera or VirtualCamera(
            display_size=self.panel.panel_size,
            sensor_size=(2000, 1220),
            sampling_ratio=0.78,
        )
        self._screen = "main"

    def read(self) -> np.ndarray:
        if self._screen == "checkerboard":
            frame = self.display.render_checkerboard()
        elif self._screen == "white":
            frame = self.display.render_white()
        else:
            frame = self.display.render()
        return self.camera.shoot(self.panel.render(frame))

    def set(self, signal: str, value: Any) -> None:
        self.display.state[signal] = value

    def show(self, screen: str) -> None:
        self._screen = screen

    def true_display_to_camera(self) -> np.ndarray:
        """Ground truth display-to-camera, for scoring a solved one."""
        dx, dy = self.panel.display_rect[0], self.panel.display_rect[1]
        offset = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        return self.camera.H_true @ offset

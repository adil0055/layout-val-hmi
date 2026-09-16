"""Stage 1 -- capture.

Two jobs: get frames, and refuse to measure a frame that is not worth
measuring.

Median-stacking N frames kills sensor noise and backlight PWM ripple for free.
The settling detector exists because needles sweep, popups slide and telltales
fade: measuring during a transition produces a number that is precise, wrong,
and indistinguishable from a real defect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol, Sequence

import cv2
import numpy as np


class SettlingTimeout(RuntimeError):
    """The screen never stopped changing.

    Raised rather than returning whatever was on screen when time ran out: a
    case that could not be measured is a failed case, not a passed one.
    """


class FrameSource(Protocol):
    """Anything that can hand out camera frames as BGR uint8 arrays."""

    def read(self) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class CameraSource:
    """A locked-down :class:`cv2.VideoCapture`.

    Every automatic setting is turned off on open.  Auto-exposure in particular
    will chase the content of the screen and change your edge profiles between
    test cases, which shows up later as an element that "moves" when it did not.
    """

    index: int | str = 0
    width: int | None = None
    height: int | None = None
    exposure: float | None = None
    """Manual exposure.  Prefer >= 10 ms, or an exact multiple of the backlight
    PWM period, so PWM banding integrates out."""

    gain: float | None = None
    white_balance: float | None = None
    warmup_s: float = 0.0
    """Both the display and the camera body expand as they heat, and that shows
    up as a slow drift of a pixel or two.  10-15 minutes before a run."""

    def __post_init__(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera {self.index!r}")
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Manual everything.  These properties are backend-dependent and may be
        # silently ignored; verify against your camera rather than trusting the
        # return value.
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if self.exposure is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        if self.gain is not None:
            self._cap.set(cv2.CAP_PROP_GAIN, self.gain)
        if self.white_balance is not None:
            self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.white_balance)
        if self.warmup_s:
            deadline = time.monotonic() + self.warmup_s
            while time.monotonic() < deadline:
                self._cap.read()

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def settings(self) -> dict[str, float]:
        """The settings actually in force, for the run record."""
        props = {
            "width": cv2.CAP_PROP_FRAME_WIDTH,
            "height": cv2.CAP_PROP_FRAME_HEIGHT,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "auto_wb": cv2.CAP_PROP_AUTO_WB,
        }
        return {name: float(self._cap.get(p)) for name, p in props.items()}

    def close(self) -> None:
        self._cap.release()


@dataclass
class DirectorySource:
    """Replay frames from disk.  Used for offline re-analysis and tests."""

    paths: Sequence[Path]
    loop: bool = True

    def __post_init__(self) -> None:
        self._paths = [Path(p) for p in self.paths]
        if not self._paths:
            raise ValueError("DirectorySource needs at least one frame")
        self._i = 0

    @classmethod
    def from_glob(cls, directory: Path | str, pattern: str = "*.png", **kw) -> "DirectorySource":
        return cls(sorted(Path(directory).glob(pattern)), **kw)

    def read(self) -> np.ndarray:
        if self._i >= len(self._paths):
            if not self.loop:
                raise RuntimeError("frame source exhausted")
            self._i = 0
        path = self._paths[self._i]
        self._i += 1
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read frame {path}")
        return img


@dataclass
class CallableSource:
    """Adapts any zero-argument callable into a :class:`FrameSource`."""

    fn: Callable[[], np.ndarray]

    def read(self) -> np.ndarray:
        return self.fn()


def capture(source: FrameSource, n: int = 15, interval_s: float = 0.0) -> list[np.ndarray]:
    """Grab ``n`` frames."""
    frames = []
    for i in range(n):
        if interval_s and i:
            time.sleep(interval_s)
        frames.append(source.read())
    return frames


def median_stack(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median of a frame burst.

    Median rather than mean: it rejects the occasional outlier frame (a PWM
    trough caught mid-exposure, a cosmic-ray pixel) instead of averaging it in.
    """
    if not frames:
        raise ValueError("median_stack needs at least one frame")
    stack = np.stack([np.asarray(f) for f in frames], axis=0)
    if stack.dtype == np.uint8:
        return np.median(stack, axis=0).astype(np.uint8)
    return np.median(stack, axis=0).astype(stack.dtype)


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR -> luma.

    Text drawn with sub-pixel anti-aliasing carries colour fringes whose weight
    depends on where the glyph falls relative to the display's RGB stripes,
    which shifts the apparent centroid per channel.  Measuring on luma averages
    that out; measuring on a single channel does not.
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class SettleResult:
    settled: bool
    frames_waited: int
    last_diff: float
    elapsed_s: float


def wait_for_settle(
    source: FrameSource,
    *,
    diff_threshold: float = 1.5,
    stable_frames: int = 5,
    timeout_s: float = 5.0,
    roi: tuple[int, int, int, int] | None = None,
    raise_on_timeout: bool = True,
) -> SettleResult:
    """Block until the screen stops changing, or fail the case.

    ``diff_threshold`` is a mean absolute frame-to-frame difference in 8-bit
    grey levels; it must sit above your sensor noise floor and below the
    smallest real animation. Measure both on a static screen before choosing it.
    """
    start = time.monotonic()
    prev = to_gray(source.read()).astype(np.float32)
    if roi:
        x, y, w, h = roi
        prev = prev[y : y + h, x : x + w]
    stable = 0
    waited = 0
    last_diff = float("inf")
    while time.monotonic() - start < timeout_s:
        cur = to_gray(source.read()).astype(np.float32)
        if roi:
            x, y, w, h = roi
            cur = cur[y : y + h, x : x + w]
        last_diff = float(np.mean(np.abs(cur - prev)))
        prev = cur
        waited += 1
        stable = stable + 1 if last_diff < diff_threshold else 0
        if stable >= stable_frames:
            return SettleResult(True, waited, last_diff, time.monotonic() - start)
    if raise_on_timeout:
        raise SettlingTimeout(
            f"screen still changing after {timeout_s:.1f}s "
            f"(last frame-to-frame diff {last_diff:.3f} > {diff_threshold})"
        )
    return SettleResult(False, waited, last_diff, time.monotonic() - start)


def settled_median(
    source: FrameSource,
    *,
    n: int = 15,
    settle: bool = True,
    **settle_kw,
) -> np.ndarray:
    """The normal way to take a measurement frame: settle, then median-stack."""
    if settle:
        wait_for_settle(source, **settle_kw)
    return median_stack(capture(source, n=n))


def estimate_noise_floor(frames: Sequence[np.ndarray]) -> float:
    """Temporal standard deviation of a static-screen burst, in grey levels.

    Use it to set the teach-in difference threshold and the settling threshold
    from the rig rather than from a guess.
    """
    stack = np.stack([to_gray(f).astype(np.float32) for f in frames], axis=0)
    return float(np.mean(np.std(stack, axis=0)))


def iter_frames(source: FrameSource, n: int) -> Iterator[np.ndarray]:
    for _ in range(n):
        yield source.read()

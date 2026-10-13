"""Setting a webcam up for a display: exposure, white balance, focus, sharpening.

A webcam left on its own exposes for the room, and a lit display in a room
is the brightest thing in it: the screen comes out blown, white strokes bloom
into each other, and a dial's ticks and numerals join into one shape. Its
white balance follows whatever else is in view, and its sharpening rings
round every white stroke. So the camera is set for the display instead, from
what the display looks like through it:

- **exposure**, so that the brightest content -- on any colour channel, so a
  red lamp's red is not clipped either -- sits just under white, and nearly
  nothing clips;
- **white balance**, so that the display's white and grey text comes out grey;
- **sharpening** off: it adds halos the measurement then has to see past;
- **gain** (ISO) to its lowest, where the camera lets a page set it, as a
  camera does in a bright scene: then the exposure needed is as long as it
  can be, with the least noise and the least flicker;
- **focus**, swept for the sharpest picture of the display, then held.

Everything is then held fixed, and the same settings are given to the camera
on every later run: a reference and a test taken at two exposures differ by
more than any layout fault.

This module only decides. The camera belongs to the browser; ``look`` puts
settings on it and returns frames taken with them (see layoutval.autorun).
The names and units are the W3C Image Capture ones the page hands over
(``MediaTrackCapabilities``): ``exposureTime`` in 100 µs, ``colorTemperature``
in kelvin.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

#: Where the brightest content should sit, on the brightest channel.
TARGET = 225.0
#: Close enough to it.
BAND = (200.0, 240.0)
#: The share of the display that may clip, at most.
CLIP_MAX = 0.0005
#: The brightest content is this percentile of the brightest channel: a
#: display is mostly dark, and its lit strokes a few percent of it.
HIGH_PCT = 99.8
#: Longest exposure used, in exposureTime's units (100 µs): one frame at 30
#: fps. Longer, and the camera drops its frame rate.
EXPOSURE_MAX = 333.0
#: Points of a focus sweep.
FOCUS_POINTS = 7


def brightest(img: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """(The brightest content's level, the share that clips), on the brightest channel.

    Over ``mask`` only, when given: the display's own content.
    """
    top = img.max(axis=2) if img.ndim == 3 else img
    if mask is not None and mask.any():
        top = top[mask]
    return float(np.percentile(top, HIGH_PCT)), float((top >= 250).mean())


def white_balance(img: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """Red over blue of what is white or grey on the display, or None if nothing is."""
    if img.ndim != 3:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    # Bright, not strongly coloured, and not clipped (clipped says nothing
    # of colour). A cast is what is being measured, and it tints the white
    # too, so "colourless" is generous; a lamp's red or blue is not.
    m = (v > 120) & (v < 245) & (hsv[..., 1] < 110)
    if mask is not None:
        m &= mask
    if m.sum() < 200:
        return None
    b, _, r = (img[..., k][m].astype(np.float64).mean() for k in range(3))
    return float(r / max(b, 1.0))


def focus_measure(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """How sharp: the variance of the Laplacian, a standard focus measure."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap[mask].var() if mask is not None and mask.any() else lap.var())


def _ratio(a: float, b: float, key: str) -> float:
    """How much more light setting ``a`` gives than ``b``."""
    return a / b if key == "exposureTime" else 2.0 ** (a - b)


def _range(caps: dict, key: str) -> tuple[float, float] | None:
    r = caps.get(key)
    if isinstance(r, dict) and "min" in r and "max" in r and r["max"] > r["min"]:
        return float(r["min"]), float(r["max"])
    return None


def _modes(caps: dict, key: str) -> list[str]:
    v = caps.get(key)
    return list(v) if isinstance(v, (list, tuple)) else []


def controls(caps: dict) -> list[str]:
    """What of this camera can be set: exposure, white balance, focus, sharpness."""
    out = []
    if (_range(caps, "exposureTime") and "manual" in _modes(caps, "exposureMode")) \
            or _range(caps, "exposureCompensation"):
        out.append("exposure")
    if _range(caps, "colorTemperature") and "manual" in _modes(caps, "whiteBalanceMode"):
        out.append("white balance")
    if _range(caps, "focusDistance") and "manual" in _modes(caps, "focusMode"):
        out.append("focus")
    if _range(caps, "sharpness"):
        out.append("sharpness")
    if _range(caps, "iso"):
        out.append("gain")
    return out


class Tuner:
    """Sets a camera up for the display it looks at.

    ``look(settings)`` puts ``settings`` on the camera (on top of what it has)
    and returns a few frames taken with them; ``region(frame)`` is the part of
    a frame that is the display (the whole frame when it is not known), and
    ``mask``, in it, the display's content -- what lit up when it came on.
    Only that is measured: a lamp in the room, or its reflection on the
    glass, that falls inside the display's outline is not the display, and a
    light that clips at every exposure would have it turned right down.
    """

    def __init__(self, caps: dict, settings: dict,
                 look: Callable[[dict], list[np.ndarray]],
                 region: Callable[[np.ndarray], np.ndarray] | None = None,
                 log: Callable[[str], None] | None = None,
                 mask: np.ndarray | None = None) -> None:
        self.caps = caps or {}
        self.now = dict(settings or {})
        self._look = look
        self.region = region or (lambda f: f)
        self.mask = mask if mask is not None and mask.any() else None
        self.log = log or (lambda m: None)
        self.steps: list[dict[str, Any]] = []

    def look(self, change: dict) -> np.ndarray:
        self.now.update(change)
        frames = [self.region(f) for f in self._look(dict(self.now))]
        return np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 1 else frames[0]

    # -- the parts ------------------------------------------------------------

    def run(self) -> dict:
        """Tune what this camera lets be tuned; the settings it ends with."""
        can = controls(self.caps)
        if not can:
            self.log("camera: nothing on it can be set from the browser")
            return {}
        held = {}
        if "sharpness" in can:
            held["sharpness"] = _range(self.caps, "sharpness")[0]
        if "gain" in can and "exposure" in can:
            held["iso"] = _range(self.caps, "iso")[0]
        if "white balance" in can and "colorTemperature" in self.now:
            held.update(whiteBalanceMode="manual", colorTemperature=self.now["colorTemperature"])
        if "focus" in can and "focusDistance" in self.now:
            held.update(focusMode="manual", focusDistance=self.now["focusDistance"])
        if held:
            self.look(held)
        if "exposure" in can:
            self.exposure()
        if "focus" in can:
            self.focus()
        if "white balance" in can:
            self.white()
        if "exposure" in can:
            self.exposure(steps=4)            # white balance moves the channels
        return {k: v for k, v in self.now.items() if k in _SETTABLE}

    def exposure(self, steps: int = 8) -> None:
        """Brightest content just under white, nearly nothing clipped."""
        if _range(self.caps, "exposureTime") and "manual" in _modes(self.caps, "exposureMode"):
            key = "exposureTime"
            lo, hi = _range(self.caps, key)
            hi = min(hi, max(lo, EXPOSURE_MAX))
            value = float(self.now.get(key, (lo + hi) / 2))
            value = min(max(value, lo), hi)
            extra = {"exposureMode": "manual"}
        else:
            key = "exposureCompensation"
            lo, hi = _range(self.caps, key)
            value = float(self.now.get(key, 0.0))
            extra = {}
        too_dark = None                         # the brightest setting known too dark
        for _ in range(steps):
            img = self.look({**extra, key: value})
            level, clip = brightest(img, self.mask)
            self.steps.append({"exposure": {key: round(value, 3), "level": round(level, 1),
                                            "clip": round(clip, 5)}})
            if clip <= CLIP_MAX and BAND[0] <= level <= BAND[1]:
                break
            if clip > CLIP_MAX or level > BAND[1]:
                # Clipped, the level says nothing of how far over: halve it,
                # or come halfway back to one known too dark.
                factor = 0.5 if too_dark is None else math.sqrt(_ratio(too_dark, value, key))
            else:
                # Below clipping a camera's response is near enough straight:
                # one step lands on the target.
                too_dark = value
                factor = min(4.0, TARGET / max(level, 1.0))
            if key == "exposureTime":
                new = min(max(value * factor, lo), hi)
            else:
                new = min(max(value + math.log2(factor), lo), hi)
            if abs(new - value) < 1e-6:
                break                          # at the end of its range
            value = new
        self.log(f"camera: exposure {key} {value:g}")

    def white(self) -> None:
        """The display's white and grey content grey: red as strong as blue."""
        lo, hi = _range(self.caps, "colorTemperature")
        t0 = float(self.now.get("colorTemperature", (lo + hi) / 2))
        r0 = white_balance(self.look({"whiteBalanceMode": "manual", "colorTemperature": t0}),
                           self.mask)
        if r0 is None:
            self.log("camera: nothing white on the display to set the white balance by")
            return
        t1 = min(hi, t0 + 800.0) if t0 + 800.0 <= hi else max(lo, t0 - 800.0)
        r1 = white_balance(self.look({"colorTemperature": t1}), self.mask)
        best_t, best_r = (t0, r0) if abs(math.log(r0)) <= abs(math.log(r1 or r0)) else (t1, r1)
        if r1 is not None and abs(r1 - r0) > 1e-3:
            # Red over blue against the setting, near enough straight here.
            t = t0 + (1.0 - r0) * (t1 - t0) / (r1 - r0)
            t = min(max(t, lo), hi)
            r = white_balance(self.look({"colorTemperature": t}), self.mask)
            if r is not None and abs(math.log(r)) < abs(math.log(best_r)):
                best_t, best_r = t, r
        self.now["colorTemperature"] = best_t
        self.look({"colorTemperature": best_t})
        self.steps.append({"white": {"colorTemperature": round(best_t), "red_over_blue": round(best_r, 3)}})
        self.log(f"camera: white balance {best_t:.0f} K, red/blue {best_r:.2f}")

    def focus(self) -> None:
        """Swept for the sharpest picture of the display, then a finer sweep round it."""
        lo, hi = _range(self.caps, "focusDistance")
        points = list(np.linspace(lo, hi, FOCUS_POINTS))
        scores = [(focus_measure(self.look({"focusMode": "manual", "focusDistance": float(p)}),
                                 self.mask), float(p)) for p in points]
        best = max(scores)[1]
        step = (hi - lo) / (FOCUS_POINTS - 1)
        finer = [best + d * step for d in (-0.5, -0.25, 0.25, 0.5) if lo <= best + d * step <= hi]
        scores += [(focus_measure(self.look({"focusDistance": float(p)}), self.mask), float(p))
                   for p in finer]
        score, best = max(scores)
        self.look({"focusDistance": best})
        self.steps.append({"focus": {"focusDistance": round(best, 4), "sharpness": round(score, 1)}})
        self.log(f"camera: focus {best:g}")


#: What the tuner sets and hands back, to be given to the camera again later.
_SETTABLE = ("exposureMode", "exposureTime", "exposureCompensation", "whiteBalanceMode",
             "colorTemperature", "focusMode", "focusDistance", "sharpness", "iso")

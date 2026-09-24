"""Glare: room light reflected off the cover glass, on top of the display.

A reflection is light added to what the display emits. The camera records the
sum, and nothing in one photograph says which part came from where. That
decides everything that can honestly be done about it:

* **Light that was added can be taken away again**, if its shape can be told
  apart from the content's. A reflection is large and smooth or large and
  flat-sided -- a window, a lamp, a ceiling panel -- and a cluster's artwork is
  small, bright strokes on a dark ground. A morphological opening larger than
  any stroke keeps the ground and the reflection and drops the artwork, so it
  estimates the added light; subtracting it *in linear light*, where light
  actually adds, removes the reflection without inventing anything.
  Done in the camera's encoded values instead, a reflection is not an offset:
  it lifts black a long way and white hardly at all, squeezing the contrast of
  whatever sits under it.

* **Light that clipped the sensor cannot be taken away.** A pixel at 255 says
  only "at least this bright"; the content under it is gone, and no algorithm
  recovers it -- not this one and not a learned one. The learned
  reflection-removal models (single-image reflection separation, 2024-2026)
  produce a *plausible* image, which is a picture of what a network expects a
  cluster to look like, and measuring it would be measuring the network. So
  they stay out of the measurement path, and a clipped element is reported as
  unmeasurable -- REVIEW, "glare" -- never as a pass or a fail.

The remedy for clipping is physical: move the camera so the reflection falls
somewhere else, shade the display, or put a polariser on the lens. An LCD's
light leaves it linearly polarised and a reflection off glass mostly is not, so
turning the polariser to pass the display's light removes much of the room.

Measured on the bench simulator, hand-held, with reflections that move between
the reference and the test shot: without this, four of six glare scenes
failed a screen with nothing wrong on it and the other two came back REVIEW;
with it, every scene that does not clip passes, and a 2 px fault still measures
1.70-1.78 px against 1.77 px with no glare at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GAMMA = 2.2
"""The phone's encoding, near enough. sRGB is a 2.2 curve with a short linear
toe; the toe is below the levels a lit cluster's ground sits at."""

CLIP_LEVEL = 250
"""At or above this in any channel, a pixel is treated as clipped. JPEG ringing
keeps saturated regions from sitting exactly at 255."""

EXCESS_MIN = 0.3
"""Added light, linear, past which a clipped pixel has lost something worth
having. A clipped pixel with reflection ``e`` on it held content somewhere
between ``1 - e`` and white; at 0.3 that is anything from 217 of 255 up, the top
fifth of an element's contrast gone. Below it the clip hides a few levels of an
already white stroke, which is what white artwork looks like to any phone,
reflection or not. (At 0.03, an unchanged bench photograph compared with itself
came back 69 elements REVIEW; at 0.16, still 7.)"""

ELEMENT_FRACTION = 0.02
"""Share of an element's box that may be glare-clipped before its measurement
stops being trusted."""

FOOTPRINT_MIN = 0.01
"""Added light, linear, that marks where a reflection was: about 30 grey levels
on a black pixel."""

_TO_LINEAR = ((np.arange(256, dtype=np.float64) / 255.0) ** GAMMA).astype(np.float32)

def to_linear(img: np.ndarray) -> np.ndarray:
    return cv2.LUT(img, _TO_LINEAR)


def to_encoded(linear: np.ndarray) -> np.ndarray:
    return (255.0 * np.clip(linear, 0.0, 1.0) ** (1.0 / GAMMA) + 0.5).astype(np.uint8)


def _top(img: np.ndarray) -> np.ndarray:
    """Largest channel, per pixel."""
    if img.ndim == 2:
        return img
    c = cv2.split(img)
    out = c[0]
    for ch in c[1:]:
        out = cv2.max(out, ch)
    return out


def bright_on_dark(frame: np.ndarray) -> bool:
    """Whether the artwork is the bright part: a night theme, as most are.

    Sparse bright strokes on a dark majority pull the mean above the median; a
    day theme's sparse dark strokes pull it below. Decided once, on the
    reference, and reused -- a reflection large enough to tip the balance on a
    later frame must not flip what counts as artwork halfway through a run.
    """
    gray = _top(frame)
    return float(np.median(gray)) <= float(gray.mean())


def _background_small(frame: np.ndarray, size_frac: float, bright: bool) -> np.ndarray:
    """The frame with its artwork taken out -- ground plus reflections -- small.

    A morphological opening -- the rolling-ball background of shading
    correction -- with a square ``size_frac`` of the frame's short side. It
    removes every bright feature narrower than that square, which is the
    artwork, and keeps everything wider exactly: the smooth hill of an
    out-of-focus lamp, and the hard edges and square corners of a window frame
    alike. (A median over the same window was tried first. It rounds a
    reflection's corners and, once blurred to hide its blockiness, softens its
    edges, and that left bright rims 50-70 levels high along every window
    frame -- straight through the elements they crossed.) For a day theme,
    dark artwork on a light ground, the same thing the other way up: a
    closing.

    Min and max commute with the encoding curve, so it is done on encoded
    values. A camera frame is worked on at reduced size first.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, 1000.0 / min(h, w))
    work = frame if scale == 1.0 else cv2.resize(
        frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    # A little smoothing first: an opening is a max of minima, and on raw
    # sensor noise it would follow the noise's troughs in blocks.
    work = cv2.GaussianBlur(work, (0, 0), 1.5)
    k = max(3, int(round(size_frac * min(work.shape[:2]))) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(work, cv2.MORPH_OPEN if bright else cv2.MORPH_CLOSE, se)


def _up(small: np.ndarray, shape: tuple[int, ...], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = shape[:2]
    if small.shape[:2] == (h, w):
        return small
    return cv2.resize(small, (w, h), interpolation=interpolation)


def _subtract(frame: np.ndarray, excess_small: np.ndarray) -> np.ndarray:
    """``frame`` less ``excess_small`` (linear, reduced size), in linear light.

    The estimate is smooth, so it is worked out small and only brought up to
    full size here; the rest is OpenCV throughout, because a phone photograph
    is fifteen million values and NumPy's version of this took over a second.
    """
    if float(excess_small.max()) < 1e-6:
        return frame
    lin = cv2.LUT(frame, _TO_LINEAR)
    e = _up(excess_small.astype(np.float32), frame.shape)
    if lin.ndim == 3 and e.ndim == 2:
        e = cv2.merge([e] * lin.shape[2])
    d = cv2.subtract(lin, e)
    np.maximum(d, 0.0, out=d)
    cv2.pow(d, 1.0 / GAMMA, dst=d)
    return cv2.convertScaleAbs(d, alpha=255.0)


def _mask(small: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return _up(small.astype(np.uint8), shape, cv2.INTER_NEAREST).astype(bool)


def _excess_own(frame: np.ndarray, size_frac: float, bright: bool | None) -> np.ndarray:
    """Light added over the frame's own ground, per pixel and channel, linear.

    With one frame alone the ground under a reflection is not known, so it is
    taken as the dim end of the background estimate -- its 10th percentile --
    and anything brighter than that and larger than an element is counted as
    added. Good enough to say *where* a reflection is and to segment a
    reference; for measuring, :func:`deglare_pair` does better.
    """
    if bright is None:
        bright = bright_on_dark(frame)
    lin = to_linear(_background_small(frame, size_frac, bright))
    floor = np.percentile(lin.reshape(-1, lin.shape[-1] if lin.ndim == 3 else 1), 10, axis=0)
    return np.maximum(lin - floor, 0.0)


def flatten(frame: np.ndarray, *, size_frac: float = 0.1,
            bright: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One frame with its reflections subtracted. Returns ``(flattened, excess)``.

    ``excess`` is per pixel, linear, the largest over the colour channels. On a
    frame with no reflection it is close to zero and this is close to a no-op.
    """
    ex = _excess_own(frame, size_frac, bright)
    return _subtract(frame, ex), _up(_top(ex), frame.shape)


@dataclass
class Pair:
    """A reference and a live frame brought down to the light they share."""

    reference: np.ndarray
    live: np.ndarray
    clipped: np.ndarray
    """Clipped under a reflection, in either frame: nothing to measure there."""
    footprint: np.ndarray
    """A reflection in either frame."""
    subtracted: float
    """The most light taken off either frame, in grey levels on black."""


def deglare_pair(reference: np.ndarray, live: np.ndarray, *, size_frac: float = 0.1,
                 bright: bool | None = None) -> Pair:
    """Subtract, from each frame, the large-scale light the other does not have.

    One frame cannot say what the ground under a reflection was; two frames of
    the same display can, because the ground is the same in both and a
    reflection only ever adds. So wherever one frame's background is brighter
    than the other's, the difference comes off, and both are left on the
    dimmer of the two. A reflection in the live frame, in the reference, or in
    both at different places is removed; one in the same place in both is left
    in both, where it cancels. (A single floor per frame was tried first, and
    left anything darker than that floor lifted by the reflection and nothing
    to take it back down.)
    """
    if reference.shape != live.shape:
        raise ValueError(f"frames differ in size: {reference.shape} and {live.shape}")
    if bright is None:
        bright = bright_on_dark(reference)
    b_ref = to_linear(_background_small(reference, size_frac, bright))
    b_live = to_linear(_background_small(live, size_frac, bright))
    common = np.minimum(b_ref, b_live)
    ex_ref, ex_live = b_ref - common, b_live - common

    # Judged on the light one frame has and the other lacks. A reflection in
    # the same place in both clips the same pixels in both, so the two are
    # compared like with like; what the pair cannot vouch for is a clip that
    # happened in one frame only.
    top_ref, top_live = _top(ex_ref), _top(ex_live)
    lost = (
        ((_top(reference) >= CLIP_LEVEL) & _mask(top_ref > EXCESS_MIN, reference.shape))
        | ((_top(live) >= CLIP_LEVEL) & _mask(top_live > EXCESS_MIN, live.shape))
    )
    taken = float(max(top_ref.max(), top_live.max()))
    return Pair(
        reference=_subtract(reference, ex_ref),
        live=_subtract(live, ex_live),
        clipped=lost,
        footprint=_mask((top_ref > FOOTPRINT_MIN) | (top_live > FOOTPRINT_MIN), live.shape),
        subtracted=round(255.0 * taken ** (1.0 / GAMMA), 1),
    )


def element_fraction(mask: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    x, y, w, h = bbox
    h_img, w_img = mask.shape[:2]
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(w_img, int(np.ceil(x + w))), min(h_img, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


GLARE = "glare"
"""Reason given to an element that glare made unmeasurable."""


def mark_unmeasurable(report, profile, mask: np.ndarray, *, values=None) -> list[str]:
    """Turn the verdict of every element glare clipped into REVIEW, "glare".

    Whatever it measured -- pass or fail -- was measured partly on pixels that
    say nothing about the display, so neither answer is one to act on. The
    measurement stays in the record. Returns the ids it changed.
    """
    from layoutval.types import Verdict, resolve_value

    hit: list[str] = []
    if not mask.any():
        return hit
    for result in report.results:
        try:
            spec = profile[result.element_id]
        except KeyError:
            continue
        box = spec.expected_bbox(resolve_value(spec, values))
        if element_fraction(mask, box) > ELEMENT_FRACTION:
            result.verdict = Verdict.REVIEW
            result.reason = GLARE
            hit.append(result.element_id)
    if hit:
        report.flag(
            GLARE, severity="review", elements=hit,
            detail=(
                "a reflection saturated the camera over these, so what is under it "
                "cannot be seen. Move the camera a little, shade the screen, or use "
                "a polarising filter, and take it again."
            ),
        )
    return hit


def set_aside_residual(report, footprint: np.ndarray) -> int:
    """Move whole-frame residual findings that sit on a reflection into a note.

    Subtracting a reflection takes its light away but not its noise -- photon
    noise grows with the light that arrived, whoever sent it -- so the dark
    ground where a reflection was comes out several times noisier than
    elsewhere, and the residual check, which compares texture, finds it every
    time. That is a finding about the room, not the display. The findings stay
    in the report under the note; the elements there were still measured.
    Returns how many it moved.
    """
    if not report.residual_findings or not footprint.any():
        return 0
    kept, moved = [], []
    for finding in report.residual_findings:
        if element_fraction(footprint, finding.bbox) > 0.5:
            moved.append(finding)
        else:
            kept.append(finding)
    if moved:
        report.residual_findings = kept
        report.flag(
            "glare_residual", severity="note",
            findings=[f.to_dict() for f in moved],
            detail=(
                f"{len(moved)} whole-frame difference(s) where a reflection was "
                "subtracted -- the reflection's own noise, most likely. Every "
                "element there was still measured."
            ),
        )
    return len(moved)


REFERENCE_AREA = 0.001
"""Share of the reference a strong reflection may cover before every result
measured against it carries a warning. Strong is :data:`EXCESS_MIN`: there a
washed-out element may not have been found at all, and one that was never found
cannot be flagged on its own. (A saturating lamp on the bench simulator clipped
14 pixels and washed a telltale out of the inventory entirely; a 2 px fault in
it then came back PASS.)"""


def reference_is_hit(mask: np.ndarray | None) -> bool:
    return mask is not None and float(mask.mean()) > REFERENCE_AREA

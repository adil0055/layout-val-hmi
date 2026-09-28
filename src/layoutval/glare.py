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

* **Light that clipped the sensor cannot be taken away** -- by this or by
  anything else. The learned reflection-removal models (single-image
  reflection separation, 2024-2026) paint in a *plausible* image, which is a
  picture of what a network expects a cluster to look like, and measuring it
  would be measuring the network; they stay out of the measurement path.
  Glare never changes a verdict on its own: the reflection is subtracted, the
  elements are measured, and how much was taken off is a note in the saved
  report.

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
"""Added light, linear, past which a clipped pixel is counted as lost to a
reflection in the saved report's note: content under it could have been
anything from 217 of 255 up."""

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


SOFT_EDGE = 0.15
"""Steepest slope of a difference over its size, below which it is out of focus.

The camera is focused on the screen; a reflection is an image of the room, a
metre or more further away, and comes out blurred. Measured: reflected room
objects 0.07-0.08, a stray mark drawn on the display 0.26-0.33 -- 0.26 with
the camera itself out of focus."""


def edge_sharpness(reference: np.ndarray, live: np.ndarray,
                   bbox: tuple[int, int, int, int], pad: int = 6) -> float:
    """How sharp the edges of what changed are: steepest slope over size of the change."""
    x, y, w, h = (int(v) for v in bbox)
    H, W = reference.shape[:2]
    sl = (slice(max(0, y - pad), min(H, y + h + pad)), slice(max(0, x - pad), min(W, x + w + pad)))
    d = _top(to_linear(live[sl])) - _top(to_linear(reference[sl]))
    d = cv2.GaussianBlur(d, (0, 0), 0.8)          # finer than this is noise, not an edge
    amp = float(np.percentile(np.abs(d), 98))
    gy, gx = np.gradient(d)
    return float(np.percentile(np.hypot(gx, gy), 98)) / max(amp, 1e-6)


def set_aside_residual(report, footprint: np.ndarray, reference: np.ndarray | None = None,
                       live: np.ndarray | None = None) -> int:
    """Move whole-frame residual findings that are the room, not the display, into a note.

    Two kinds. Where a large reflection was subtracted, its light goes but its
    photon noise stays, and the dark ground there comes out several times
    noisier than elsewhere. And a reflected room object smaller than the
    opening's square -- a shelf, a lamp, a window -- is not subtracted at all,
    and shows as a patch of added light; but it is out of focus, where anything
    the display draws is sharp (:data:`SOFT_EDGE`). On glared bench photographs
    those came back REVIEW with every element passing. The findings stay in the
    report under the note; the elements there were still measured. Returns how
    many it moved.
    """
    if not report.residual_findings:
        return 0
    kept, moved = [], []
    for finding in report.residual_findings:
        lit = footprint.any() and element_fraction(footprint, finding.bbox) > 0.5
        soft = (reference is not None and live is not None
                and edge_sharpness(reference, live, finding.bbox) < SOFT_EDGE)
        (moved if lit or soft else kept).append(finding)
    if moved:
        report.residual_findings = kept
        report.flag(
            "glare_residual", severity="note",
            findings=[f.to_dict() for f in moved],
            detail=(
                f"{len(moved)} whole-frame difference(s) that are reflections -- "
                "noise where one was subtracted, or an out-of-focus object from "
                "the room. Every element there was still measured."
            ),
        )
    return len(moved)


FADE_MAX = 0.6
"""Contrast an element under glare may keep, as a share of the reference's,
before what it shows can no longer be told from something else."""


def _contrast(patch: np.ndarray) -> float:
    """Contrast at the scale of strokes: a light blur against a wider one."""
    from layoutval.capture import to_gray

    g = to_gray(patch).astype(np.float32)
    return float((cv2.GaussianBlur(g, (0, 0), 0.7) - cv2.GaussianBlur(g, (0, 0), 4.0)).std())


def recheck_under_glare(report, profile, reference: np.ndarray, live: np.ndarray,
                        footprint: np.ndarray, clipped: np.ndarray, *, values=None) -> list[str]:
    """Check identity again, on detail alone, for elements a reflection lay over.

    A compact, bright reflection -- a lamp rather than a window -- is only
    partly subtracted: the top of its hill is narrower than the opening can
    follow. What it leaves is a smooth slope of light across whatever sits
    under it, and correlation, which forgives an even change of brightness but
    not a slope, reads a telltale on that slope as a different telltale. Over a
    good screen that was a FAIL, which is the reflection deciding the verdict.

    So an element whose identity check failed under a reflection is compared
    again at the position that was measured, on its fine detail only -- each
    patch less a blur of itself, which a smooth slope cannot survive and a
    stroke does -- and without the pixels the reflection clipped, which carry
    nothing (two pixels of margin for the bloom round a clip). The same element
    matches; a different symbol still differs in its strokes and stays a
    failure. Needs 30% of the element unclipped. Returns the ids it changed.
    """
    from layoutval.capture import to_gray
    from layoutval.measure import subpixel_crop
    from layoutval.types import resolve_value
    from layoutval.verdict import WRONG_CONTENT, evaluate

    if not footprint.any():
        return []
    lost = cv2.dilate(clipped.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    lit = footprint.astype(np.uint8) * 255
    changed: list[str] = []
    unconfirmed: list[str] = []
    unmeasured: list[str] = []
    skipped: list[int] = []
    for i, result in enumerate(report.results):
        m = result.measurement
        if result.reason != WRONG_CONTENT or m.dx is None or m.dy is None:
            continue
        try:
            spec = profile[result.element_id]
        except KeyError:
            continue
        x, y, w, h = spec.expected_bbox(resolve_value(spec, values))
        here, there = (x, y, w, h), (x + m.dx, y + m.dy, w, h)
        if not ((subpixel_crop(lit, here) > 0).any() or (subpixel_crop(lit, there) > 0).any()):
            continue
        keep = ~((subpixel_crop(lost, here) > 0) | (subpixel_crop(lost, there) > 0))
        ref_patch, live_patch = subpixel_crop(reference, here), subpixel_crop(live, there)
        if keep.mean() >= 0.3 and keep.sum() >= 30:
            sigma = max(1.5, min(w, h) / 6.0)

            def detail(patch: np.ndarray) -> np.ndarray:
                g = to_gray(patch).astype(np.float32)
                return (g - cv2.GaussianBlur(g, (0, 0), sigma)).astype(np.float64)[keep]

            a, b = detail(ref_patch), detail(live_patch)
            a, b = a - a.mean(), b - b.mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            score = float((a * b).sum()) / den if den > 0 else 0.0
            if score >= result.tolerance.identity_min:
                m.zncc = score
                m.method += " (detail, under glare)"
                report.results[i] = evaluate(spec, m)
                changed.append(spec.id)
                continue
        # Faded past telling: a reflection plus the phone's own processing --
        # local tone mapping flattens contrast where it is bright, noise
        # reduction smears what is faint -- can leave an element a ghost of
        # itself. Measured on two bench photographs, such elements kept 5-57% of
        # their contrast, and at that point the same symbol and a different one
        # score alike: no comparison can say which it is. A wrong symbol that
        # is still visible keeps all its contrast (161-172%, with or without a
        # lamp on it) and stays a failure. A faded one is judged on where it is,
        # which is still measured to a fraction of a pixel, and the report says
        # what it is showing could not be confirmed.
        fade = _contrast(live_patch) / max(_contrast(ref_patch), 1e-6)
        if fade < FADE_MAX:
            m.method += f" (faded to {fade:.0%} by glare; identity unconfirmed, zncc {m.zncc:.2f})"
            m.zncc = None
            judged = evaluate(spec, m)
            # Faded far enough, the position goes too -- and the measurement
            # says so itself: its two estimators stop agreeing. Out of
            # tolerance with them apart, it is not measured at all; out of
            # tolerance with them together, it moved, and it fails.
            apart = m.estimator_disagreement_px is None or m.estimator_disagreement_px > 1.0
            if judged.verdict.value != "PASS" and apart:
                skipped.append(i)
                unmeasured.append(spec.id)
            else:
                report.results[i] = judged
                unconfirmed.append(spec.id)
            changed.append(spec.id)
    for i in reversed(skipped):
        del report.results[i]
    if unconfirmed or unmeasured:
        parts = []
        if unconfirmed:
            parts.append(f"{len(unconfirmed)} were checked for position only")
        if unmeasured:
            parts.append(f"{len(unmeasured)} could not be measured and were skipped")
        report.flag(
            "identity_unconfirmed", severity="note",
            elements=unconfirmed, skipped=unmeasured,
            detail=(
                f"{len(unconfirmed) + len(unmeasured)} element(s) under glare were too "
                f"faded to confirm what they show: {' and '.join(parts)}. Retake "
                "without the reflection to check them fully."
            ),
        )
    return changed

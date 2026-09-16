"""Stage 4 -- measuring where each element actually landed.

Two metrics, always, not one:

* **Phase correlation** tells you *how far it moved*.
* **ZNCC** tells you *whether it is still the same thing*.

Running only one is how an element rendering the wrong symbol gets reported as
a position failure, and whoever picks up the defect wastes a day in layout code.

Search is always confined to the expected box dilated by the element's margin --
never the whole screen, which is how you match a similar-looking element
somewhere else entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np

from layoutval.capture import to_gray
from layoutval.types import ElementKind, ElementSpec, Measurement

Rect = tuple[int, int, int, int]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def subpixel_peak(res: np.ndarray, x: int, y: int) -> tuple[float, float] | None:
    """Quadratic fit around an integer correlation peak.

    Returns ``None`` when the peak sits on the border of the response map.  That
    is not a formality: a peak on the edge of the search window means the element
    may have moved further than the window can measure.  The magnitude is
    unknown, so the caller must report a failure, not a pass and not a clamped
    value.
    """
    h, w = res.shape[:2]
    if not (0 < x < w - 1 and 0 < y < h - 1):
        return None
    cx0, cx, cx1 = float(res[y, x - 1]), float(res[y, x]), float(res[y, x + 1])
    cy0, cy1 = float(res[y - 1, x]), float(res[y + 1, x])
    denom_x = cx0 - 2.0 * cx + cx1
    denom_y = cy0 - 2.0 * cx + cy1
    dx = 0.5 * (cx0 - cx1) / denom_x if abs(denom_x) > 1e-12 else 0.0
    dy = 0.5 * (cy0 - cy1) / denom_y if abs(denom_y) > 1e-12 else 0.0
    # A parabola fitted through a flat or noisy neighbourhood can extrapolate
    # outside the sample it was fitted to; that is not a sub-pixel estimate.
    if abs(dx) > 1.0 or abs(dy) > 1.0:
        return float(x), float(y)
    return float(x) + dx, float(y) + dy


@lru_cache(maxsize=32)
def _hann(w: int, h: int) -> np.ndarray:
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


def phase_shift(
    ref_patch: np.ndarray, live_patch: np.ndarray
) -> tuple[tuple[float, float], float]:
    """Sub-pixel translation from ``ref_patch`` to ``live_patch``.

    Sign convention, pinned by :func:`tests.test_measure.test_phase_sign`:
    content at ``u`` in ``ref_patch`` appears at ``u + (dx, dy)`` in
    ``live_patch``.

    A Hann window is applied unconditionally.  Without it, spectral leakage at
    the patch edges corrupts the peak -- and it corrupts it in a way that looks
    like a plausible small shift rather than like a failure.
    """
    if ref_patch.shape != live_patch.shape:
        raise ValueError(
            f"phase_shift needs equal-sized patches, got {ref_patch.shape} and "
            f"{live_patch.shape}"
        )
    a = np.float32(to_gray(ref_patch))
    b = np.float32(to_gray(live_patch))
    h, w = a.shape[:2]
    (dx, dy), response = cv2.phaseCorrelate(a, b, _hann(w, h))
    return (float(dx), float(dy)), float(response)


def zncc_match(
    search_area: np.ndarray, template: np.ndarray
) -> tuple[tuple[float, float], float, bool]:
    """Locate ``template`` inside ``search_area`` by zero-mean normalised
    cross-correlation.

    Returns ``((x, y), score, peak_on_border)`` where ``(x, y)`` is the
    sub-pixel top-left of the match in ``search_area`` coordinates.  ZNCC is
    used rather than plain correlation because it absorbs a global gain and
    offset change, which is exactly what backlight dimming does.  It does *not*
    absorb a gamma change -- teach each theme separately.
    """
    s = np.float32(to_gray(search_area))
    t = np.float32(to_gray(template))
    if s.shape[0] < t.shape[0] or s.shape[1] < t.shape[1]:
        raise ValueError(
            f"search area {s.shape} smaller than template {t.shape}; "
            "widen search_margin_px or check the expected box"
        )
    res = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    refined = subpixel_peak(res, int(loc[0]), int(loc[1]))
    if refined is None:
        return (float(loc[0]), float(loc[1])), float(score), True
    return refined, float(score), False


def is_degenerate(patch: np.ndarray, *, min_std: float = 1.0) -> bool:
    """True when a patch carries no structure to correlate against.

    A uniform patch returns a perfect ZNCC against any other uniform patch, so
    an unguarded identity check on one says "same thing" about two completely
    different colours.
    """
    return float(np.std(np.float32(to_gray(patch)))) < min_std


def centroid_from_mask(mask: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """Intensity-weighted centroid and area of a binary mask.

    For an isolated luminous telltale this beats correlation: the element is
    bright on a dark background, so the first moment is dominated by the element
    itself and averages over every lit pixel.
    """
    m = cv2.moments(mask.astype(np.float32), binaryImage=False)
    if m["m00"] <= 0:
        return None
    return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])), float(m["m00"])


def angle_difference_deg(a: float, b: float) -> float:
    """``a - b`` wrapped into (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d + 360.0 if d <= -180.0 else d


@dataclass
class NeedlePose:
    """Where a needle is, as the two independent quantities it actually has."""

    pivot: tuple[float, float]
    angle_deg: float
    length_px: float
    hub_found: bool
    """Whether the pivot came from the hub or from the mask's tail extreme.  The
    fallback is biased outwards by roughly the hub radius, so it is recorded
    rather than hidden."""


def needle_pose(
    mask: np.ndarray,
    expected_pivot: tuple[float, float],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    hub_width_factor: float = 1.6,
) -> NeedlePose | None:
    """Pivot and angle of a needle, as two independent quantities.

    A needle rotates; it is not a shift-invariant patch, and phase correlation on
    one returns nonsense.  The principal axis of the mask gives the direction, and
    the two ends of that axis are the tail and the tip.

    The pivot is taken from the **hub**, not from the tail extreme.  Every real
    needle has a hub, and the tail extreme sits a hub-radius beyond the true
    pivot -- a systematic offset of several pixels that a repeatability study
    correctly reports as bias but that no amount of averaging removes.  The hub
    is detected as the run of bins at the tail end whose perpendicular width
    exceeds ``hub_width_factor`` times the shaft's, and the pivot is the centroid
    of the mask after eroding the shaft away.  With no hub (a bare shaft) the
    tail extreme is used and ``hub_found`` is False, so the consumer can see that
    the number carries that bias.

    Angle is degrees from the +x axis, counter-clockwise positive as seen on the
    display (display y runs downwards, so the sign is flipped relative to raw
    array coordinates).
    """
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 16:
        return None
    pts = pts.reshape(-1, 2).astype(np.float64)
    pts[:, 0] += origin[0]
    pts[:, 1] += origin[1]

    centre = pts.mean(axis=0)
    centred = pts - centre
    # Principal axis: the eigenvector of the scatter matrix with the larger
    # eigenvalue.  Equivalent to the long side of minAreaRect, but it uses every
    # mask pixel rather than the convex hull, so a single stray pixel cannot
    # swing it.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    perp = np.array([-axis[1], axis[0]])

    t = centred @ axis
    r = np.abs(centred @ perp)
    exp = np.asarray(expected_pivot, dtype=np.float64)
    if np.linalg.norm((centre + axis * t.max()) - exp) < np.linalg.norm(
        (centre + axis * t.min()) - exp
    ):
        # Orient so t increases from tail towards tip.
        axis, t = -axis, -t
    tip_pt = centre + axis * float(t.max())

    n_bins = 12
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 4.0:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    widths = np.array(
        [r[idx == b].max() if np.any(idx == b) else 0.0 for b in range(n_bins)]
    )
    # The shaft is the outer half, which the hub never reaches into.
    shaft = float(np.median(widths[n_bins // 2 :])) or 1.0

    hub_bins: list[int] = []
    for b in range(n_bins):
        if widths[b] > shaft * hub_width_factor:
            hub_bins.append(b)
        else:
            break

    tail_pt = centre + axis * float(t.min())
    pivot, hub_found = tail_pt, False
    if hub_bins:
        hub_radius = float(r[np.isin(idx, hub_bins)].max())
        # Erode away the shaft and take what is left.  The shaft is narrower
        # than the hub by construction, so an erosion between the two widths
        # leaves the hub alone; its centroid is then the pivot, with no
        # contribution from the shaft to pull it towards the tip and no
        # dependence on a single extreme pixel.
        er = int(min(max(round(shaft) + 1, 2), max(2, int(hub_radius) - 2)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))
        hub_pts = cv2.findNonZero(cv2.erode(mask, kernel))
        if hub_pts is not None and len(hub_pts) >= 4:
            hp = hub_pts.reshape(-1, 2).astype(np.float64)
            pivot = np.array([hp[:, 0].mean() + origin[0], hp[:, 1].mean() + origin[1]])
            hub_found = True
        else:
            # Erosion took everything: fall back to the tail extreme stepped in
            # by one hub radius, which is the same point derived from the two
            # least robust statistics available.
            pivot = tail_pt + axis * hub_radius
            hub_found = True

    v = tip_pt - pivot
    return NeedlePose(
        pivot=(float(pivot[0]), float(pivot[1])),
        angle_deg=float(math.degrees(math.atan2(-v[1], v[0]))),
        length_px=float(np.linalg.norm(v)),
        hub_found=hub_found,
    )


# --------------------------------------------------------------------------
# cropping helpers
# --------------------------------------------------------------------------


def _clip_rect(rect: Rect, shape: tuple[int, ...]) -> Rect:
    h, w = shape[:2]
    x, y, rw, rh = rect
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x) + int(rw)), min(h, int(y) + int(rh))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def crop(img: np.ndarray, rect: Rect) -> np.ndarray:
    x, y, w, h = _clip_rect(rect, img.shape)
    return img[y : y + h, x : x + w]


def subpixel_crop(img: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Extract a patch whose top-left pixel centre is exactly at ``(x, y)``.

    ``cv2.getRectSubPix`` interpolates, so the returned patch is anchored at the
    fractional coordinate rather than at the nearest integer one.  Everything
    downstream then measures displacement against that exact anchor.
    """
    x, y, w, h = rect
    pw, ph = int(round(w)), int(round(h))
    if pw < 1 or ph < 1:
        return np.empty((0, 0), dtype=img.dtype)
    centre = (x + (pw - 1) / 2.0, y + (ph - 1) / 2.0)
    return cv2.getRectSubPix(img, (pw, ph), centre)


def search_rect(
    spec: ElementSpec,
    value: float | None,
    shape: tuple[int, ...],
    *,
    anchor: tuple[float, float] | None = None,
    size: tuple[float, float] | None = None,
) -> Rect:
    """Expected patch position dilated by the search margin, clipped to the frame.

    Never the whole screen: an unbounded search is how a template matches a
    similar-looking element somewhere else entirely and reports a confident,
    enormous, meaningless displacement.
    """
    x, y = anchor if anchor is not None else spec.expected_bbox(value)[:2]
    w, h = size if size is not None else spec.bbox[2:]
    m = spec.search_margin_px
    return _clip_rect((round(x - m), round(y - m), round(w + 2 * m), round(h + 2 * m)), shape)


# --------------------------------------------------------------------------
# per-kind measurement
# --------------------------------------------------------------------------


@dataclass
class MaskSpec:
    """How to turn a display-space crop into a binary element mask.

    Defaults to a luminance threshold, which is right for a lit telltale on a
    dark background.  Pass ``hsv_range`` to reuse the HSV masks already built for
    telltale colour validation.
    """

    luma_threshold: int = 60
    hsv_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    close_kernel: int = 3

    @classmethod
    def from_config(cls, cfg: dict | None) -> "MaskSpec":
        if not cfg:
            return cls()
        hsv = cfg.get("hsv_range")
        return cls(
            luma_threshold=int(cfg.get("luma_threshold", 60)),
            hsv_range=(tuple(hsv[0]), tuple(hsv[1])) if hsv else None,
            close_kernel=int(cfg.get("close_kernel", 3)),
        )

    def build(self, patch: np.ndarray) -> np.ndarray:
        if self.hsv_range is not None and patch.ndim == 3:
            lo, hi = self.hsv_range
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        else:
            _, mask = cv2.threshold(
                to_gray(patch), self.luma_threshold, 255, cv2.THRESH_BINARY
            )
        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask


def measure_translation(
    reference: np.ndarray,
    live: np.ndarray,
    spec: ElementSpec,
    *,
    value: float | None = None,
    template: np.ndarray | None = None,
    max_estimator_disagreement_px: float = 0.5,
) -> Measurement:
    """Coarse-to-fine translation measurement for a shift-invariant element.

    ZNCC over the search area locates the element to about a third of a pixel
    and answers the identity question; phase correlation, run between the
    template and a live patch cut at that coarse location, refines the residual
    to well under a tenth.  Both numbers are kept in the record -- a high ZNCC
    with a large offset means the element is correct but misplaced, and a low
    ZNCC means something else is wrong, which is a different defect reported
    differently.
    """
    m = Measurement(element_id=spec.id, method="zncc")
    ex, ey, ew, eh = spec.expected_bbox(value)
    m.expected_centre = (ex + ew / 2.0, ey + eh / 2.0)

    # The template has its own anchor: taught templates are cut to the ink, which
    # normally sits inside the design's layout box.  Measure displacement against
    # the anchor the template was actually cut at, then report it as the
    # element's displacement -- the two differ by a constant that is authoring
    # information, not a defect.
    if template is None:
        if spec.position.kind != "static":
            m.error = (
                f"{spec.id} moves with its signal but has no stored template; the "
                "reference cannot supply one at an arbitrary state"
            )
            return m
        ax, ay, aw, ah = ex, ey, ew, eh
        # Sub-pixel extraction, not an integer crop.  A template cut at round(ex)
        # but differenced against ex carries up to half a pixel of invented
        # offset -- larger than everything this pipeline is trying to measure,
        # and different per element, so it looks like structure rather than a bug.
        template = subpixel_crop(reference, (ax, ay, aw, ah))
    else:
        ax, ay = spec.template_origin(value)
        ah, aw = template.shape[:2]
    if template.size == 0 or min(template.shape[:2]) < 3:
        m.error = "template is empty or smaller than 3 px; check the expected box"
        return m
    if is_degenerate(template):
        m.error = (
            "template has no structure to correlate against (uniform patch); "
            "neither its position nor its identity can be measured"
        )
        return m

    th, tw = template.shape[:2]
    sx, sy, sw, sh = search_rect(spec, value, live.shape, anchor=(ax, ay), size=(tw, th))
    area = crop(live, (sx, sy, sw, sh))
    if area.shape[0] < th or area.shape[1] < tw:
        m.error = (
            f"search area {area.shape[1]}x{area.shape[0]} is smaller than the "
            f"template {tw}x{th}; the expected box is at or past the frame edge"
        )
        return m

    (px, py), zncc, on_border = zncc_match(area, template)
    m.zncc = zncc
    m.peak_on_search_border = on_border

    coarse_x, coarse_y = sx + px, sy + py
    m.zncc_dx, m.zncc_dy = coarse_x - ax, coarse_y - ay

    # Fine stage: cut both patches at the coarse location, so the residual the
    # phase correlator sees is sub-pixel -- the regime it is good in -- and pad
    # them with context, so the element's edges are not sitting under the Hann
    # window's taper where they contribute almost nothing.
    ix, iy = int(round(coarse_x)), int(round(coarse_y))
    pad_x, pad_y = spec.context()
    ref_ctx = subpixel_crop(
        reference, (ax - pad_x, ay - pad_y, tw + 2 * pad_x, th + 2 * pad_y)
    )
    live_ctx = crop(live, (ix - pad_x, iy - pad_y, tw + 2 * pad_x, th + 2 * pad_y))

    observed_x, observed_y = coarse_x, coarse_y
    if (
        ref_ctx.shape[:2] == live_ctx.shape[:2]
        and min(live_ctx.shape[:2]) >= 8
        and ix - pad_x >= 0
        and iy - pad_y >= 0
    ):
        (fx, fy), response = phase_shift(ref_ctx, live_ctx)
        m.phase_response = response
        m.phase_dx, m.phase_dy = ix + fx - ax, iy + fy - ay
        m.estimator_disagreement_px = math.hypot(
            m.phase_dx - m.zncc_dx, m.phase_dy - m.zncc_dy
        )
        # The coarse peak is within half a pixel of the truth by construction, so
        # the two estimates must agree to about that.  When they do not, one of
        # them is outside its validity regime -- most often phase correlation,
        # because the element's edges are sitting under the Hann window's taper
        # or because content entered the patch that was not in it before.  Both
        # numbers stay in the record either way; neither is silently discarded,
        # and the method string says which one became the verdict's input.
        if spec.estimator == "phase":
            observed_x, observed_y = ix + fx, iy + fy
            m.method = "phase"
        elif m.estimator_disagreement_px > max_estimator_disagreement_px:
            m.method = "zncc(phase-disagrees)"
        else:
            m.method = "zncc+phase"

    m.dx = observed_x - ax
    m.dy = observed_y - ay
    m.observed_centre = (m.expected_centre[0] + m.dx, m.expected_centre[1] + m.dy)
    return m


def measure_telltale(
    reference: np.ndarray,
    live: np.ndarray,
    spec: ElementSpec,
    *,
    value: float | None = None,
    template: np.ndarray | None = None,
    mask_spec: MaskSpec | None = None,
) -> Measurement:
    """A telltale: correlation for position, the mask for presence and area.

    The usual advice is to measure an isolated luminous telltale by the centroid
    of its mask, and on repeatability grounds that looks right -- a mask centroid
    is very stable frame to frame.  Measured against a *known* displacement it is
    not: thresholding re-quantises the anti-aliased edge differently at every
    sub-pixel phase, and this implementation's linearity study puts that bias
    near a pixel on a small glyph, against a static-screen sigma of 0.15 px.

    So correlation reports the position, and the mask reports what only it can:
    whether the element is lit at all, and how much of it there is.  Set
    ``estimator="centroid"`` to reverse that, and run the linearity study before
    doing so.
    """
    mask_spec = mask_spec or MaskSpec()
    m = measure_translation(reference, live, spec, value=value, template=template)

    sx, sy, sw, sh = search_rect(spec, value, live.shape)
    live_area, ref_area = crop(live, (sx, sy, sw, sh)), crop(reference, (sx, sy, sw, sh))
    if live_area.size == 0:
        return m

    live_c = centroid_from_mask(mask_spec.build(live_area))
    ref_c = centroid_from_mask(mask_spec.build(ref_area))
    if live_c is None:
        if ref_c is not None:
            m.element_absent = True
            m.error = "no lit pixels in the search area, but the reference has them"
        return m
    (lcx, lcy), area = live_c
    m.area_px = area / 255.0
    if ref_c is not None:
        # Both centroids are taken over the same window, so any asymmetry in the
        # mask -- an anti-aliased edge, a glow -- biases both sides identically
        # and cancels out of the difference.
        (rcx, rcy), _ = ref_c
        m.centroid_dx, m.centroid_dy = lcx - rcx, lcy - rcy
        if spec.estimator == "centroid":
            m.dx, m.dy = m.centroid_dx, m.centroid_dy
            m.observed_centre = (sx + lcx, sy + lcy)
            m.method = "centroid"
    return m


def measure_needle(
    reference: np.ndarray,
    live: np.ndarray,
    spec: ElementSpec,
    *,
    value: float | None = None,
    mask_spec: MaskSpec | None = None,
    expected_angle_deg: float | None = None,
) -> Measurement:
    """Pivot position and needle angle, measured independently.

    A needle at the right angle about the wrong pivot is a real and separate
    defect, so the two quantities get their own tolerances and their own reasons
    in the verdict.
    """
    mask_spec = mask_spec or MaskSpec()
    m = Measurement(element_id=spec.id, method="needle")
    if spec.pivot is None:
        m.error = "needle element has no pivot; set ElementSpec.pivot in display coords"
        return m

    sx, sy, sw, sh = search_rect(spec, value, live.shape)
    live_area = crop(live, (sx, sy, sw, sh))
    if live_area.size == 0:
        m.error = "expected box lies outside the frame"
        return m

    pose = needle_pose(mask_spec.build(live_area), spec.pivot, origin=(sx, sy))
    if pose is None:
        m.error = "needle mask empty or too small to fit an axis"
        return m
    (pvx, pvy), angle = pose.pivot, pose.angle_deg
    if not pose.hub_found:
        m.method = "needle(no-hub)"

    m.pivot_observed = (pvx, pvy)
    m.observed_centre = (pvx, pvy)
    m.expected_centre = tuple(spec.pivot)
    m.dx = pvx - spec.pivot[0]
    m.dy = pvy - spec.pivot[1]
    m.angle_deg = angle

    expected = (
        expected_angle_deg if expected_angle_deg is not None else spec.expected_angle(value)
    )
    if expected is None:
        # No modelled angle for this state: fall back to the reference capture so
        # the check still means something, and record where it came from.
        ref_pose = needle_pose(
            mask_spec.build(crop(reference, (sx, sy, sw, sh))), spec.pivot, origin=(sx, sy)
        )
        if ref_pose is not None:
            expected = ref_pose.angle_deg
            m.method = "needle(ref-angle)"
    if expected is not None:
        m.expected_angle_deg = float(expected)
        m.d_angle_deg = angle_difference_deg(angle, float(expected))
    return m


MeasureFn = Callable[..., Measurement]

_DISPATCH: dict[ElementKind, MeasureFn] = {
    ElementKind.ICON: measure_translation,
    ElementKind.TEXT: measure_translation,
    ElementKind.REGION: measure_translation,
    ElementKind.TELLTALE: measure_telltale,
    ElementKind.NEEDLE: measure_needle,
}


def measure_element(
    reference: np.ndarray,
    live: np.ndarray,
    spec: ElementSpec,
    *,
    value: float | None = None,
    template: np.ndarray | None = None,
    mask_spec: MaskSpec | None = None,
) -> Measurement:
    """Measure one element with the method its kind calls for."""
    fn = _DISPATCH[spec.kind]
    kwargs: dict = {"value": value}
    if fn is measure_translation:
        if spec.kind is ElementKind.TEXT:
            # Sub-pixel anti-aliasing puts colour fringes on glyph edges; measure
            # text on luma so the per-channel centroid shift cancels.
            reference, live = to_gray(reference), to_gray(live)
            if template is not None:
                template = to_gray(template)
        kwargs["template"] = template
    elif fn is measure_telltale:
        kwargs["template"] = template
        kwargs["mask_spec"] = mask_spec
    else:
        kwargs["mask_spec"] = mask_spec
    return fn(reference, live, spec, **kwargs)

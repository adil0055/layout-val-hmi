"""Catching what you did not model.

Per-element checks only find problems with elements you knew about.  After
measuring everything on the list, diff the rectified frame against the reference
and look at what is left over.

This check is **advisory, always**.  Pixel-level comparison of camera captures is
noisy enough that a hard threshold either fires constantly or is set so loose it
catches nothing.  Its job is to surface the stray artefact, the element nobody
taught and the wrong z-order -- things a per-element measurement passes cleanly.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from layoutval.capture import to_gray
from layoutval.measure import crop, is_degenerate, zncc_match
from layoutval.types import (
    ElementResult,
    ElementSpec,
    Measurement,
    ResidualFinding,
    Verdict,
    resolve_value,
)


def _rect_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[int, int, int, int] | None:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))


def residual_check(
    reference: np.ndarray,
    live: np.ndarray,
    specs: list[ElementSpec] | None = None,
    *,
    values: dict[str, float] | None = None,
    dissimilarity_threshold: float = 0.45,
    min_area_px: int = 64,
    blur_sigma: float = 1.0,
    max_findings: int = 20,
) -> tuple[float, list[ResidualFinding]]:
    """SSIM residual over the whole rectified frame.

    Returns the global SSIM score and the regions that stand out, each annotated
    with which known elements it overlaps.  A finding that overlaps a known
    element usually means that element changed content rather than moved; a
    finding that overlaps nothing is the interesting case.

    A light blur is applied first: it suppresses the single-pixel disagreement
    that camera resampling always produces, without touching the
    tens-of-pixels-wide artefacts this check exists to find.
    """
    ref = to_gray(reference).astype(np.float32)
    lv = to_gray(live).astype(np.float32)
    if ref.shape != lv.shape:
        raise ValueError(f"residual_check needs equal shapes, got {ref.shape} and {lv.shape}")
    if blur_sigma > 0:
        ref = cv2.GaussianBlur(ref, (0, 0), blur_sigma)
        lv = cv2.GaussianBlur(lv, (0, 0), blur_sigma)

    score, diff_map = ssim(ref, lv, data_range=255.0, full=True)
    dissim = (1.0 - diff_map).astype(np.float32)

    mask = (dissim > dissimilarity_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    findings: list[ResidualFinding] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x, y, w, h = (int(stats[i, k]) for k in range(4))
        region = dissim[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i]
        overlaps = []
        for spec in specs or []:
            v = resolve_value(spec, values)
            if _rect_overlap((x, y, w, h), spec.expected_bbox(v)) is not None:
                overlaps.append(spec.id)
        findings.append(
            ResidualFinding(
                bbox=(x, y, w, h),
                mean_dissimilarity=float(region.mean()),
                area_px=area,
                overlaps=overlaps,
            )
        )

    findings.sort(key=lambda f: f.area_px * f.mean_dissimilarity, reverse=True)
    return float(score), findings[:max_findings]


def dissimilarity_heatmap(
    reference: np.ndarray, live: np.ndarray, *, blur_sigma: float = 1.0
) -> np.ndarray:
    """A colourised 1-SSIM map to attach to the review flag.

    The scalar is not enough to triage with: reviewers need to see *where*.
    """
    ref = to_gray(reference).astype(np.float32)
    lv = to_gray(live).astype(np.float32)
    if blur_sigma > 0:
        ref = cv2.GaussianBlur(ref, (0, 0), blur_sigma)
        lv = cv2.GaussianBlur(lv, (0, 0), blur_sigma)
    _, diff_map = ssim(ref, lv, data_range=255.0, full=True)
    dissim = np.clip((1.0 - diff_map) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(dissim, cv2.COLORMAP_INFERNO)


def check_occlusions(
    reference: np.ndarray,
    live: np.ndarray,
    specs: list[ElementSpec],
    *,
    values: dict[str, float] | None = None,
    identity_min: float | None = None,
) -> list[ElementResult]:
    """Explicit z-order assertions for element pairs that overlap by design.

    Two elements each in the correct position, with the wrong one drawn on top,
    passes every per-element position check you can write.  The residual map is
    the automatic catch; this is the deliberate one, driven by
    :attr:`ElementSpec.occludes`, and unlike the residual map it produces a real
    verdict because the author asserted the relationship on purpose.
    """
    by_id = {s.id: s for s in specs}
    results: list[ElementResult] = []
    for spec in specs:
        for other_id in spec.occludes:
            other = by_id.get(other_id)
            if other is None:
                continue
            v_a = resolve_value(spec, values)
            v_b = resolve_value(other, values)
            region = _rect_overlap(spec.expected_bbox(v_a), other.expected_bbox(v_b))
            check_id = f"{spec.id}>{other_id}"
            m = Measurement(element_id=check_id, method="occlusion", dx=0.0, dy=0.0)
            if region is None:
                m.error = (
                    f"{spec.id} is declared to occlude {other_id} but their expected "
                    "boxes do not overlap; the assertion cannot be evaluated"
                )
                results.append(
                    ElementResult(check_id, Verdict.REVIEW, "occlusion_not_applicable", m, spec.tolerance, spec.source)
                )
                continue

            ref_patch, live_patch = crop(reference, region), crop(live, region)
            if min(ref_patch.shape[:2]) < 3:
                continue
            if is_degenerate(ref_patch):
                # Two flat patches correlate perfectly whatever their colours, so
                # the assertion would pass on any wrong answer.  Say so instead.
                m.error = (
                    f"the overlap between {spec.id} and {other_id} is a flat patch; "
                    "z-order cannot be judged by correlation here"
                )
                results.append(
                    ElementResult(check_id, Verdict.REVIEW, "occlusion_indeterminate",
                                  m, spec.tolerance, spec.source)
                )
                continue
            try:
                _, score, _ = zncc_match(live_patch, ref_patch)
            except ValueError:
                continue
            m.zncc = score
            threshold = identity_min if identity_min is not None else spec.tolerance.identity_min
            if score < threshold:
                results.append(
                    ElementResult(check_id, Verdict.FAIL, "z_order", m, spec.tolerance, spec.source)
                )
            else:
                results.append(
                    ElementResult(check_id, Verdict.PASS, None, m, spec.tolerance, spec.source)
                )
    return results

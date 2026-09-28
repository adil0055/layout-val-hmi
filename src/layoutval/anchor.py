"""The last step of a hand-held pose: taken from the elements themselves.

The pose from the whole screen (ECC, seeded by matched features) is fitted to
brightness, and anything that changes brightness without moving the screen
pulls it: a haze of room light across the glass, a reflection that slid when
the camera moved, a silhouette of the person holding the phone. On a simulated
bench with the user's own cluster in a laptop window, a haze over the glass and
the phone turned 4 degrees between the shots, that pull left a good screen with
25-29 elements failing, up to 5 px out -- the error growing towards one side,
which is what a pose that is slightly wrong does.

The elements are what the measurement is about, they lie on the display's own
plane, and each is located separately, on its own detail, re-checked under
glare. So after a first measurement, the pose is corrected by the one smooth
mapping that best explains where the elements were found -- a homography
across the display, fitted with RANSAC so that elements that really moved are
left out of it and still report their movement. Then everything is measured
again under the corrected pose.

**What it can and cannot absorb.** A pose error is smooth across the whole
display, and one element that moved is not: with many elements, no homography
bends to fit one of them without leaving the others behind. With few elements
it could, so the model is reduced as the count falls -- a homography from 20
elements, an affine map from 8 -- and below that nothing is changed. A fault
where every element moved together is absorbed, as it already was by any
hand-held pose; ``--edges`` checks that.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from layoutval.types import ElementKind, RunReport, resolve_value

INLIER_PX = 1.0
"""An element within this of the fitted mapping, display px, supports it."""

MIN_HOMOGRAPHY = 20
MIN_AFFINE = 8

#: Kinds whose measurement is a plain shift. A needle turns and a bar fills.
SHIFTING = {ElementKind.ICON, ElementKind.TEXT, ElementKind.TELLTALE, ElementKind.REGION}


@dataclass
class Correction:
    G: np.ndarray
    """Display -> display: where each point of the layout was found. The pose
    corrected is ``H @ G``."""
    used: int
    inliers: int
    largest_px: float
    """Largest correction at any element used, display px."""


def can_anchor(profile) -> bool:
    """Whether ``profile`` has enough elements that shift to anchor a pose on."""
    return sum(1 for s in profile if s.kind in SHIFTING) >= MIN_AFFINE


def element_correction(report: RunReport, profile, values=None) -> Correction | None:
    """The smooth mapping that explains where the elements were found, or None."""
    specs = {s.id: s for s in profile}
    src, dst = [], []
    for r in report.results:
        m = r.measurement
        spec = specs.get(r.element_id)
        if (spec is None or spec.kind not in SHIFTING or m.dx is None or m.dy is None
                or m.element_absent or r.reason == "missing_or_displaced"):
            continue
        x, y, w, h = spec.expected_bbox(resolve_value(spec, values))
        src.append((x + w / 2, y + h / 2))
        dst.append((x + w / 2 + m.dx, y + h / 2 + m.dy))
    n = len(src)
    if n < MIN_AFFINE:
        return None
    src_a, dst_a = np.float64(src), np.float64(dst)
    if n >= MIN_HOMOGRAPHY:
        G, mask = cv2.findHomography(src_a, dst_a, cv2.RANSAC, INLIER_PX,
                                     maxIters=5000, confidence=0.999)
        if G is not None and mask is not None and mask.sum() >= 4:
            keep = mask.ravel() > 0
            G, _ = cv2.findHomography(src_a[keep], dst_a[keep], 0)
    else:
        A, mask = cv2.estimateAffine2D(src_a, dst_a, method=cv2.RANSAC,
                                       ransacReprojThreshold=INLIER_PX,
                                       maxIters=5000, confidence=0.999)
        G = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])
        if G is not None and mask is not None and mask.sum() >= 3:
            keep = mask.ravel() > 0
            A, _ = cv2.estimateAffine2D(src_a[keep], dst_a[keep], method=cv2.LMEDS)
            G = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])
    if G is None or mask is None:
        return None
    inliers = int(mask.sum())
    if inliers < max(MIN_AFFINE - 2, 0.6 * n):
        return None                       # no one mapping explains most of them
    moved = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), G).reshape(-1, 2) - src_a
    return Correction(G=G, used=n, inliers=inliers,
                      largest_px=float(np.linalg.norm(moved, axis=1).max()))

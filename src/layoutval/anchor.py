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

**A group that moved together.** A homography can bend: near the edge of the
display, where nothing else pins it, it can follow part of a group of elements
that moved together -- a whole gauge drawn 4 px to the right -- and leave the
rest of the group as the only failures. On the user's bench that flagged the
right half of the left gauge and passed its left half; on two bench photographs
with the left gauge moved 3-4 px, 66% of its elements were flagged and 86
elements elsewhere were flagged instead, some of them with the camera not moved
at all. Counting inliers cannot tell those apart, because the bent mapping
explains *more* elements than the right one. So each candidate mapping is
scored by the elements it explains plus the largest set of the rest that share
one displacement of at least GROUP_SHIFT_PX: under the right mapping the moved
gauge is one such set, all of it; under a bent one its displacements differ
from element to element. Scored that way, 345 of 345 moved-gauge elements were
flagged, and 20 of 639 elsewhere -- mostly neighbours the planted move touched;
of a 3 px move of the left third of the layout, 82% of its elements were
flagged where 41% had been.
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

GROUP_MIN = 3
"""Elements that must share one displacement to count as a group that moved."""

GROUP_SHIFT_PX = 2.0
"""Smallest displacement, display px, for such a group to count: twice INLIER_PX,
so a group stands clear of the mapping's own scatter."""

HYPOTHESES = 3000
"""Candidate mappings tried, each from four elements chosen at random."""

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
    grouped: int = 0
    """Elements left out of the mapping because they moved together, as a group."""


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
        found = _grouped_homography(src_a, dst_a)
        if found is None:
            return None
        G, inliers, grouped = found
        if inliers + grouped < max(MIN_AFFINE - 2, 0.6 * n) or inliers < MIN_HOMOGRAPHY:
            return None                   # no one mapping explains most of them
        moved = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), G).reshape(-1, 2) - src_a
        return Correction(G=G, used=n, inliers=inliers, grouped=grouped,
                          largest_px=float(np.linalg.norm(moved, axis=1).max()))
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


def _project(Hs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points through each of a stack of homographies: (k, n, 2)."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    q = np.einsum("kij,nj->kni", Hs, p)
    return q[..., :2] / q[..., 2:3]


def _explained(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inliers, the largest group of the rest that moved together), as masks."""
    norm = np.linalg.norm(residuals, axis=1)
    inlier = norm < INLIER_PX
    group = np.zeros_like(inlier)
    rest = np.flatnonzero(~inlier & (norm >= GROUP_SHIFT_PX))
    if len(rest) >= GROUP_MIN:
        r = residuals[rest]
        together = np.linalg.norm(r[:, None] - r[None], axis=2) < INLIER_PX
        centre = int(together.sum(axis=1).argmax())
        if together[centre].sum() >= GROUP_MIN:
            group[rest[together[centre]]] = True
    return inlier, group


def _grouped_homography(src: np.ndarray, dst: np.ndarray):
    """RANSAC for the mapping, scored by what it explains including a moved group.

    The winner is refitted on its inliers together with the group, the group
    taken as one rigid block with a shift of its own: nothing else may pin the
    side of the screen it is on, and its own shape does. Returns
    (G, inliers, grouped), or None.
    """
    n = len(src)
    rng = np.random.default_rng(0)
    samples = np.array([rng.choice(n, 4, replace=False) for _ in range(HYPOTHESES)])
    stack = np.full((HYPOTHESES, 3, 3), np.nan)
    for k, s in enumerate(samples):
        try:
            stack[k] = cv2.getPerspectiveTransform(src[s].astype(np.float32),
                                                   dst[s].astype(np.float32))
        except cv2.error:
            pass                          # three of the four in a line
    with np.errstate(all="ignore"):
        residuals = _project(stack, src) - dst[None]
    finite = np.isfinite(residuals).all(axis=(1, 2))
    counts = (np.linalg.norm(residuals, axis=2) < INLIER_PX).sum(axis=1)
    best, best_key = None, (-1, -1)
    for k in np.flatnonzero(finite & (counts >= 0.3 * n)):
        inlier, group = _explained(residuals[k])
        key = (int(inlier.sum() + group.sum()), int(inlier.sum()))
        if key > best_key:
            best, best_key = k, key
    if best is None:
        return None
    keep, group = _explained(residuals[best])
    shift = -np.median(residuals[best][group], axis=0) if group.any() else np.zeros(2)
    G = None
    for _ in range(20):
        if keep.sum() < 4:
            return None
        G, _ = cv2.findHomography(np.concatenate([src[keep], src[group]]),
                                  np.concatenate([dst[keep], dst[group] - shift]), 0)
        if G is None:
            return None
        residual = _project(G[None], src)[0] - dst
        now_keep, now_group = _explained(residual)
        # residual = mapped - found, so the group's own shift is minus its residual
        now_shift = -np.median(residual[now_group], axis=0) if now_group.any() else np.zeros(2)
        settled = ((now_keep == keep).all() and (now_group == group).all()
                   and np.abs(now_shift - shift).max() < 0.01)
        keep, group, shift = now_keep, now_group, now_shift
        if settled:
            break
    return G, int(keep.sum()), int(group.sum())

"""Whether the whole layout moved, measured against the screen's own edges.

A hand-held pose is solved from what the screen shows. That is what lets the
camera move between shots, and it has one blind spot: a fault in which every
element moved together looks exactly like the camera having moved, and the pose
absorbs it. A whole screen drawn 2 px to the right passes.

The display's edges do not move with what is drawn on it. So each edge is
carried from the reference photograph into the test photograph through the
solved pose, and compared with what is actually there: a strip across the edge,
sampled in both photos, and the shift that lines the two up. The pose follows
the drawing, so an edge that has moved against it means the drawing moved
against the edge.

**Compared, not found.** Fitting each edge to a clean intensity step works on
the simulator and not on a laptop: the lit area's boundary there is dark grey
against a black bezel, with a highlight on the glass beside it, and no side of
two bench photographs gave a clean step. Lining up the same strip in two photos
needs no step -- whatever is there, the same thing is there twice. Measured on
those two photographs, with the camera moved and the drawing shifted 2 px
inside its edges: 1.97-2.08 px on every side, and under 0.06 px unshifted.

**Only the edge and beyond.** The strip reaches barely inside the edge, or the
drawing near it pulls the answer towards zero: taken symmetrically, one side of
a bench photograph read 0.35 px for the 2 px shift.

**What it relies on.** That the edge is in the plane of the pixels -- the lit
area's own boundary, or trim flush with it. Trim or a cover glass standing a
few millimetres proud moves against the pixels when the camera moves, by about
that depth times the change in viewing angle. So it is opt-in (``--edges``),
and where two opposite sides disagree it says it could not check.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from layoutval.capture import to_gray
from layoutval.displayfind import propose_display_corners

AGREE_PX = 0.5
"""Largest disagreement between opposite sides, display px, for a shift to count."""

SCATTER_MAX = 0.3
"""Largest scatter (median absolute deviation, photo px) along one side."""


@dataclass
class SideOffset:
    offset: float
    """How far the edge sits outward of where the pose puts it, photo px."""
    scatter: float
    normal: np.ndarray
    middle: np.ndarray


@dataclass
class LayoutShift:
    dx: float | None
    dy: float | None
    """The drawing against the edges, display px; None where no side was usable."""

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx or 0.0, self.dy or 0.0))


def reference_corners(undistorted: np.ndarray) -> np.ndarray | None:
    """The display's four corners in the reference photo, top-left first."""
    proposal = propose_display_corners(undistorted)
    return None if proposal is None else np.asarray(proposal.corners, np.float64)


def _sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.remap(gray, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)


def _shift(r: np.ndarray, v: np.ndarray, lo: int, hi: int, max_lag: int) -> float | None:
    """s with v[x + s] ~ r[x] over r[lo:hi], in samples: ZNCC, parabolic peak."""
    rr = r[lo:hi] - r[lo:hi].mean()
    nr = float(np.linalg.norm(rr))
    if nr < 1e-6:
        return None
    scores = np.full(2 * max_lag + 1, -1.0)
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        ll = v[lo + lag:hi + lag] - v[lo + lag:hi + lag].mean()
        nl = float(np.linalg.norm(ll))
        if nl > 1e-6:
            scores[k] = float(rr @ ll) / (nr * nl)
    k = int(np.argmax(scores))
    if k in (0, len(scores) - 1) or scores[k] < 0.7:
        return None
    a, b, c = scores[k - 1], scores[k], scores[k + 1]
    den = a - 2 * b + c
    return (k - max_lag) + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)


def edge_offsets(reference: np.ndarray, live: np.ndarray, corners: np.ndarray,
                 warp: np.ndarray, *, groups: int = 10, per_group: int = 8,
                 step: float = 0.5) -> list[SideOffset | None]:
    """Each side's offset in ``live`` from where ``warp`` carries it (top, right, bottom, left).

    ``corners`` are in ``reference``; ``warp`` maps the reference photo onto
    the live one. None for a side with nothing to line up on, or too little of
    it inside both photos.
    """
    g_ref = to_gray(reference).astype(np.float32)
    g_live = to_gray(live).astype(np.float32)
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    centre = corners.mean(axis=0)
    out: list[SideOffset | None] = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        length = float(np.linalg.norm(b - a))
        along = (b - a) / length
        normal = np.array([-along[1], along[0]])
        if normal @ ((a + b) / 2 - centre) < 0:
            normal = -normal                                   # outwards
        reach = step * round(max(8.0, 0.012 * length) / step)
        offs = np.arange(-reach, reach + step / 2, step)
        lag = int(round(reach / 2 / step))
        lo = max(lag, int(np.searchsorted(offs, -0.15 * reach)))
        hi = len(offs) - lag
        ts = np.linspace(0.1, 0.9, groups * per_group)
        pts = (a + along * (length * ts)[:, None])[:, None, :] + normal * offs[None, :, None]
        ref_p = _sample(g_ref, pts)
        live_p = _sample(g_live, cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2), warp).reshape(pts.shape))
        inside = (ref_p.min(axis=1) >= 0) & (live_p.min(axis=1) >= 0)
        shifts = []
        for g in range(groups):
            sl = slice(g * per_group, (g + 1) * per_group)
            keep = inside[sl]
            if keep.sum() < per_group // 2:
                continue
            r, v = ref_p[sl][keep].mean(axis=0), live_p[sl][keep].mean(axis=0)
            if np.abs(np.diff(r[lo:hi])).max() < 2.0 * step:
                continue                                        # nothing to line up on
            s = _shift(r, v, lo, hi, lag)
            if s is not None:
                shifts.append(s * step)
        if len(shifts) < 4:
            out.append(None)
            continue
        med = float(np.median(shifts))
        out.append(SideOffset(med, float(np.median(np.abs(np.array(shifts) - med))),
                              normal, (a + b) / 2))
    return out


def layout_shift(H_ref: np.ndarray, sides: list[SideOffset | None],
                 drawn: tuple[float, float]) -> LayoutShift:
    """The drawing against the edges, in display px.

    ``H_ref`` maps display to the reference photo; ``drawn`` is how far the
    elements themselves were measured to have moved against the pose (their
    median, so one faulty element does not count).
    """
    inv = np.linalg.inv(H_ref)
    moved: list[float | None] = []
    for i, side in enumerate(sides):
        if side is None or side.scatter > SCATTER_MAX:
            moved.append(None)
            continue
        ends = np.array([[side.middle], [side.middle + side.normal * side.offset]])
        d = np.diff(cv2.perspectiveTransform(ends, inv).reshape(2, 2), axis=0)[0]
        moved.append(float(d[1] if i % 2 == 0 else d[0]))       # top/bottom: y

    def axis(a: float | None, b: float | None, drawn_by: float) -> float | None:
        got = [v for v in (a, b) if v is not None]
        if not got or (len(got) == 2 and abs(got[0] - got[1]) > AGREE_PX):
            return None
        return drawn_by - float(np.mean(got))

    return LayoutShift(dx=axis(moved[1], moved[3], drawn[0]),
                       dy=axis(moved[0], moved[2], drawn[1]))

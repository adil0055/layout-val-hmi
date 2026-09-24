"""Find a display's four corners in an ordinary photograph of it.

This proposes the corners that the hand-marking route otherwise asks somebody
to tap. It is deterministic, needs nothing from the cluster, and it is a
*proposal*: the phone shows the four dots and a person accepts or drags them,
so a wrong guess costs a second rather than a measurement.

How it works, and why each part is there -- every rule below replaced one that
was measured to fail on a real bench photograph:

**Edges come from line segments, not from thresholds.** A display's boundary
against its bezel is the lowest-contrast edge in the picture, dark on dark, and
its sign is not even consistent: on a black cluster in a glossy black bezel the
bezel reflects the room and ends up *brighter* than the screen along the
bottom, and darker along the top. What is consistent is that the boundary is a
long straight line -- the panel's black mask -- and a line segment detector
finds it on every side. Local contrast equalisation first, because without it
that edge came back in fragments after a modest exposure change.

**Each side is chosen on its own, innermost first.** A photograph of a display
holds several nested rectangles of about the right shape: the screen, a window
inside it, a title bar, the bezel's outer edge, the lid, a keyboard. Scoring
whole quadrilaterals picked a lid-and-keyboard quad; so does preferring large
ones, and preferring small ones picks the title bar.

**The rule that separates them is that nothing lit sits on a bezel.** For each
candidate line, look outward toward the next line out -- for the display's own
edge that stretch *is* the bezel -- and ask whether anything lit is there. The
display edge is the innermost strong line with a dark stretch beyond it. A
title-bar line fails because the clock and icons sit between it and the next
line out; an interior line fails because content lies beyond it; the lid edge
passes too, but is not innermost. "Lit" means small bright marks (text, ticks,
icons) from a top-hat filter, which by construction ignores a smooth glare
patch.

**Measured only over the display itself.** Along each line, only the stretch
between the two neighbouring sides counts. The detected line runs on past the
display's corners into the desk and keyboard, and measured along its whole
length the true bottom edge of one photograph failed on keyboard keys that
were not beside it at all.

Measured: on two bench photographs, each warped 30 ways (perspective, rotation,
scale 0.65-1.0, exposure 0.6-1.3x), with and without a synthetic glare patch,
120 of 120 land within 1% of the image diagonal, median worst corner 9-10 px on
2000-2600 px images. On the simulated cluster in its bezel it agrees with the
sub-pixel border fit to 0.2-0.3 px.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

#: The search runs at this size. Plenty for choosing lines; the chosen lines
#: are then re-fitted at full resolution.
WORK_SIZE = 1000

#: Largest fraction of a line's positions allowed to show lit content in the
#: band beyond it before it stops counting as a bezel.
SPILL_TOLERANCE = 0.02

#: Fraction of a side, between its neighbours, that a line has to cover.
MIN_COVERAGE = 0.45


@dataclass
class _Line:
    p: np.ndarray
    d: np.ndarray
    segs: list[tuple[np.ndarray, np.ndarray]]
    support: float


@dataclass
class CornerProposal:
    """Four corners in the photograph, top-left first, and how sure it is."""

    corners: np.ndarray
    confident: bool
    sides: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_fractions(self, shape: tuple[int, ...]) -> list[list[float]]:
        h, w = shape[:2]
        return [[float(x) / w, float(y) / h] for x, y in self.corners]

    def to_dict(self, shape: tuple[int, ...]) -> dict[str, Any]:
        return {
            "corners": self.as_fractions(shape),
            "confident": self.confident,
            "sides": self.sides,
        }


# --------------------------------------------------------------------------
# lines
# --------------------------------------------------------------------------


def _segments(gray: np.ndarray) -> np.ndarray:
    """Line segments, (N, 4). LSD where OpenCV has it, a Hough fallback where not."""
    work = cv2.GaussianBlur(gray, (3, 3), 0)
    work = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(work)
    try:
        segs = cv2.createLineSegmentDetector().detect(work)[0]
    except (cv2.error, AttributeError):
        # LSD was absent from OpenCV 4.1 to 4.5.0. Probabilistic Hough on an
        # edge map is coarser but finds the same long straight boundaries.
        edges = cv2.Canny(work, 30, 90)
        segs = cv2.HoughLinesP(edges, 1, np.pi / 720, 40,
                               minLineLength=int(0.02 * max(gray.shape)), maxLineGap=6)
    if segs is None:
        return np.zeros((0, 4), np.float32)
    return np.asarray(segs, np.float32).reshape(-1, 4)


def _families(gray: np.ndarray) -> tuple[list[_Line], list[_Line]]:
    h, w = gray.shape
    hor, ver = [], []
    for x1, y1, x2, y2 in _segments(gray):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 0.02 * max(w, h):
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        seg = (np.array([x1, y1]), np.array([x2, y2]), length)
        if angle < 30 or angle > 150:
            hor.append(seg)
        elif abs(angle - 90) < 30:
            ver.append(seg)
    return _merge(hor, horizontal=True), _merge(ver, horizontal=False)


def _merge(segs, *, horizontal: bool, dist: float = 4.0, dangle: float = 4.0) -> list[_Line]:
    """Collinear segments into lines, fitted once at the end."""
    def orient(d: np.ndarray) -> np.ndarray:
        return -d if (horizontal and d[0] < 0) or (not horizontal and d[1] < 0) else d

    lines: list[_Line] = []
    for a, b, length in sorted(segs, key=lambda s: -s[2]):
        d = orient((b - a) / np.linalg.norm(b - a))
        for line in lines:
            n = np.array([-line.d[1], line.d[0]])
            angle = np.degrees(np.arccos(np.clip(abs(line.d @ d), -1, 1)))
            if angle < dangle and abs((a - line.p) @ n) < dist and abs((b - line.p) @ n) < dist:
                line.segs.append((a, b))
                line.support += length
                break
        else:
            lines.append(_Line((a + b) / 2, d, [(a, b)], length))
    for line in lines:
        pts = np.array([p for s in line.segs for p in (s[0], s[1], (s[0] + s[1]) / 2)],
                       np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        line.p, line.d = np.array([x0, y0]), orient(np.array([vx, vy]))
    return lines


def _intersect(a: _Line, b: _Line) -> np.ndarray | None:
    m = np.column_stack([a.d, -b.d])
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    t = np.linalg.solve(m, b.p - a.p)
    return a.p + t[0] * a.d


def _crossing(line: _Line, axis: int, at: float) -> float:
    """Where a line crosses the column (axis=1) or row (axis=0) ``at``."""
    if axis == 1:
        return float(line.p[1] + (at - line.p[0]) * line.d[1] / line.d[0])
    return float(line.p[0] + (at - line.p[1]) * line.d[0] / line.d[1])


def _extent(line: _Line) -> tuple[float, float]:
    t = [(p - line.p) @ line.d for s in line.segs for p in s]
    return (min(t), max(t))


def _between(line: _Line, a: _Line, b: _Line, inset: float = 0.06) -> tuple[float, float]:
    pa, pb = _intersect(line, a), _intersect(line, b)
    if pa is None or pb is None:
        return _extent(line)
    lo, hi = sorted([(pa - line.p) @ line.d, (pb - line.p) @ line.d])
    margin = (hi - lo) * inset
    return (lo + margin, hi - margin)


def _coverage(line: _Line, span: tuple[float, float]) -> float:
    lo, hi = span
    if hi <= lo:
        return 0.0
    intervals = sorted(
        tuple(sorted(((s[0] - line.p) @ line.d, (s[1] - line.p) @ line.d)))
        for s in line.segs)
    covered, cur = 0.0, None
    for a, b in intervals:
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        if cur is None or a > cur[1]:
            if cur:
                covered += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        covered += cur[1] - cur[0]
    return covered / (hi - lo)


# --------------------------------------------------------------------------
# the bezel test
# --------------------------------------------------------------------------


def _lit_marks(gray: np.ndarray) -> np.ndarray:
    """Small bright marks -- text, ticks, icons -- and not smooth glare."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_TOPHAT, kernel)
    tophat = tophat.astype(np.float32)
    # Half the 99th percentile: a ratio, so it follows exposure. Measured on
    # two bench photographs, screen text peaks at 146-159 and reflections on a
    # glossy bezel at up to 54, with this landing at 57 between them. A
    # threshold set from the image's noise instead (18) was tried, and read the
    # bezel's reflections as lit: success on the warped-photograph set fell from
    # 120/120 to 101/120, one of them confidently wrong.
    threshold = max(18.0, 0.5 * float(np.percentile(tophat, 99.0)))
    return (tophat > threshold).astype(np.uint8)


def _spill(marks: np.ndarray, line: _Line, outward: np.ndarray, lo: int, hi: int,
           span: tuple[float, float]) -> float:
    """Fraction of positions along the line with a lit mark in the band beyond it.

    Sampled at every pixel along the span. Sampling every tenth one aliased
    against text: strokes are a pixel or two wide, so a clock in a desktop bar
    hit one sample in eighty and a title-bar line passed as a bezel.
    """
    if hi < lo:
        return 0.0
    h, w = marks.shape
    samples = max(40, int(abs(span[1] - span[0])))
    ts = np.linspace(span[0], span[1], samples)
    base = line.p[None, :] + ts[:, None] * line.d[None, :]
    depths = np.arange(lo, hi + 1)
    pts = np.round(base[:, None, :] + depths[None, :, None] * outward[None, None, :])
    pts = pts.reshape(-1, 2).astype(int)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    hit = np.zeros(len(pts))
    hit[inside] = marks[pts[inside, 1], pts[inside, 0]]
    return float(np.mean(hit.reshape(samples, len(depths)).max(axis=1) > 0))


def _pick(lines: list[_Line], side: str, marks: np.ndarray, centre: np.ndarray,
          span_of, gate) -> tuple[_Line | None, dict[str, float]]:
    h, w = marks.shape
    horizontal = side in ("top", "bottom")
    axis = 1 if horizontal else 0
    at = centre[0] if horizontal else centre[1]
    dim = w if horizontal else h
    outward_sign = -1 if side in ("top", "left") else +1
    where = lambda line: _crossing(line, axis, at)  # noqa: E731

    everyone = [line for line in lines if line.support >= 0.06 * dim]
    candidates = sorted(
        (line for line in everyone
         if (where(line) - centre[axis]) * outward_sign > 0 and gate(line)),
        key=lambda line: abs(where(line) - centre[axis]),
    )
    tried = []
    for line in candidates:
        here = where(line)
        # A neighbour closer than 8 px is the far side of the same thin
        # feature, and a band that narrow would pass anything.
        beyond = [abs(where(o) - here) for o in everyone
                  if (where(o) - here) * outward_sign > 8]
        gap = min(beyond + [0.06 * (h if horizontal else w)])
        n = np.array([-line.d[1], line.d[0]])
        outward = n if n[axis] * outward_sign > 0 else -n
        # Only the inner 60% of the way out: the middle of a bezel is reliably
        # bezel, its far edge is where the desk shows through under a hinge.
        spill = _spill(marks, line, outward, 3, max(6, int(0.6 * gap)), span_of(line))
        tried.append((line, spill))
        if spill <= SPILL_TOLERANCE:
            return line, {"spill": spill, "clean": 1.0}
    if not tried:
        return None, {}
    line, spill = min(tried, key=lambda x: x[1])
    return line, {"spill": spill, "clean": 0.0}


# --------------------------------------------------------------------------
# full-resolution fit
# --------------------------------------------------------------------------


def _refit(gray: np.ndarray, corners: np.ndarray, band: float = 6.0,
           max_angle: float = 2.0) -> np.ndarray:
    """Re-fit each chosen side on the untouched full-resolution image.

    Equalisation finds faint edges but shifts where they sit, so the side is
    *chosen* on the equalised image and *located* on the original. The fit may
    only nudge a corner -- it corrects a position, it does not get to choose a
    different edge.
    """
    h, w = gray.shape
    fitted = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        x0 = int(max(0, min(a[0], b[0]) - band - 2))
        x1 = int(min(w, max(a[0], b[0]) + band + 2))
        y0 = int(max(0, min(a[1], b[1]) - band - 2))
        y1 = int(min(h, max(a[1], b[1]) + band + 2))
        pts = []
        if x1 - x0 > 4 and y1 - y0 > 4:
            try:
                segs = cv2.createLineSegmentDetector().detect(gray[y0:y1, x0:x1])[0]
            except (cv2.error, AttributeError):
                segs = None
            for sx1, sy1, sx2, sy2 in (segs.reshape(-1, 4) if segs is not None else []):
                p1 = np.array([sx1 + x0, sy1 + y0])
                p2 = np.array([sx2 + x0, sy2 + y0])
                sd = p2 - p1
                length = np.linalg.norm(sd)
                if length < 20:
                    continue
                if np.degrees(np.arccos(min(1.0, abs(sd @ d) / length))) > max_angle:
                    continue
                if max(abs((p1 - a) @ n), abs((p2 - a) @ n)) > band:
                    continue
                pts.extend(p1 + t * sd for t in np.linspace(0, 1, max(2, int(length // 10))))
        if len(pts) < 8:
            fitted.append((a, d))
            continue
        vx, vy, px, py = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER,
                                     0, 0.01, 0.01).ravel()
        fitted.append((np.array([px, py]), np.array([vx, vy])))

    def cross(l1, l2):
        m = np.column_stack([l1[1], -l2[1]])
        t = np.linalg.solve(m, l2[0] - l1[0])
        return l1[0] + t[0] * l1[1]

    # side i runs from corner i to corner i+1, so corner i = side(i-1) x side(i)
    try:
        refit = np.array([cross(fitted[(i - 1) % 4], fitted[i]) for i in range(4)])
    except np.linalg.LinAlgError:
        return corners
    if np.linalg.norm(refit - corners, axis=1).max() > 0.005 * np.hypot(h, w):
        return corners
    return refit


# --------------------------------------------------------------------------


def propose_display_corners(frame: np.ndarray, *, passes: int = 3) -> CornerProposal | None:
    """The display's four corners in ``frame``, or None if no display is found.

    ``confident`` is False when any side had to fall back to its least-bad line
    rather than one with a clean bezel beyond it -- the proposal is still made,
    because a wrong guess is cheap to drag, but the page says to check it.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = WORK_SIZE / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    marks = _lit_marks(small)
    hor, ver = _families(small)
    centre = np.array([w / 2.0, h / 2.0])
    families = {"top": hor, "bottom": hor, "left": ver, "right": ver}
    dims = {"top": w, "bottom": w, "left": h, "right": h}
    neighbours = {"top": ("left", "right"), "bottom": ("left", "right"),
                  "left": ("top", "bottom"), "right": ("top", "bottom")}

    # Pass 1 knows nothing about the display yet, so only a loose floor on
    # line length, to find approximate neighbours.
    picked = {side: _pick(fam, side, marks, centre, _extent,
                          lambda line, side=side: line.support >= 0.15 * dims[side])
              for side, fam in families.items()}
    # Later passes judge each line on the display's own terms: how much of the
    # side between its neighbours it covers, measured only there. A fixed
    # fraction of the *image* rejected a true edge when the display was small
    # in frame.
    for _ in range(passes - 1):
        if any(line is None for line, _ in picked.values()):
            return None
        prev = {side: line for side, (line, _) in picked.items()}
        spans = {side: (lambda line, side=side: _between(
            line, prev[neighbours[side][0]], prev[neighbours[side][1]]))
            for side in families}
        picked = {side: _pick(fam, side, marks, centre, spans[side],
                              lambda line, side=side: _coverage(line, spans[side](line))
                              >= MIN_COVERAGE)
                  for side, fam in families.items()}
        if all(picked[s][0] is prev[s] for s in picked):
            break
    if any(line is None for line, _ in picked.values()):
        return None

    lines = {side: line for side, (line, _) in picked.items()}
    corners = [_intersect(lines["top"], lines["left"]), _intersect(lines["top"], lines["right"]),
               _intersect(lines["bottom"], lines["right"]),
               _intersect(lines["bottom"], lines["left"])]
    if any(c is None for c in corners):
        return None
    corners = np.array(corners) / scale
    quad = corners.astype(np.float32)
    if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 0.04 * gray.size:
        return None
    corners = _refit(gray, corners)

    sides = {}
    for side, (line, info) in picked.items():
        span = _between(line, lines[neighbours[side][0]], lines[neighbours[side][1]])
        # Plain floats: these go to the phone as JSON, which has no float32.
        sides[side] = {"coverage": round(float(_coverage(line, span)), 3),
                       "spill": round(float(info.get("spill", 1.0)), 3),
                       "clean": bool(info.get("clean", 0.0))}
    confident = all(s["clean"] and s["coverage"] >= MIN_COVERAGE for s in sides.values())
    return CornerProposal(corners=corners, confident=confident, sides=sides)

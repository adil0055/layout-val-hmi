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

**Every element gets a vote.** Which set is the pose and which moved is
decided by count, so every element has to be in the count. Measured as usual,
an element is searched for 8 px round where the first pose puts it -- and the
first pose can lock onto the moved group: on a webcam shot with the right of
the screen hazy, the sharp left dial was what the matched features and ECC
followed, the rest of the screen sat 8 px or more off, came back "missing or
displaced" and had no vote, and the moved dial passed while everything else
failed. So the first correction is taken from every element looked for up to
VOTE_MARGIN_PX away (:func:`far_correction`), and only then are the elements
measured as usual. With the first pose locked onto a dial moved 6-12 px, 2 of
12 runs on two bench photographs had flipped; none do now, all 434 of the dial's
elements flagged and 36 elsewhere where there had been 165.

**Few elements, several things moved.** A sparse cluster of 32 elements with
its speed band moved 4 px and the speed digits and gear letter 3 px left 19
unmoved -- one short of the 20 a homography was allowed from -- so no
correction was made and the pose stayed where ECC put it, pulled 1-2 px
towards what moved: half the band passed. With the gear letter left alone, 20
were left and it worked, which is how moving one element changed the verdict
on others. Now an affine map, which cannot bend, is fitted when fewer than 20
are left; more than one group may move, by different amounts; and the
mapping must be fitted to more elements than any group holds, or a mapping
lined up on what moved scores as well as the right one. On hazy webcam-size
frames of a bench cluster cut to 32 elements, with the first pose pulled
towards three moved things, 35 of 171 unmoved elements were flagged; 4 now.

**Two answers that fit alike.** With a mounted webcam on a sparse cluster, a
reference taken once and the same faulted screen captured again and again,
some captures came back inside out: the moved speed band passing, the unmoved
elements beside it failing. Nothing is learnt between captures; a reflection
growing on the glass was enough. ECC, pulled by the sharp band, bent the first
pose towards it, so the band read 0 and the rest carried a stretch; at 0.51
camera px per display px each element is placed to 0.5-1 px. Then the band
(11) moved 4 px and the view stretched 0.3% with the 4 elements beside it
moved the other way fit every element alike, and counting inliers took the
second. So: the best 40 distinct candidates are refitted before they are
compared; every element's own shift, and the screen exactly where it was
taught, are candidates too; and of explanations within TIE_ELEMENTS of the
best, the one nearest where the screen was taught is taken -- if that is
itself among them, so a camera or lid that really moved is not held to it.
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

MAX_GROUPS = 3
"""At most this many groups. A fault moves a thing or a few; a camera that moved
leaves a smooth field of offsets, which, carved into enough small groups,
"explains" every element -- and with the screen where it was taught it then
looked like the answer, on a phone turned 4 degrees with one block moved 2 px."""

GROUP_SHIFT_PX = 2.0
"""Smallest displacement, display px, for such a group to count: twice INLIER_PX,
so a group stands clear of the mapping's own scatter."""

HYPOTHESES = 3000
"""Candidate mappings tried, each from four elements chosen at random."""

REFINED = 40
"""How many of the best distinct candidates are refitted before they are compared."""

TIE_ELEMENTS = 2
"""Explanations this many elements apart are as good as each other; the one
that leaves the screen nearest where it was taught is taken."""

AT_HOME_PX = 1.0
"""A pose within this, display px on average, of the taught one leaves the
screen where it was."""

VOTE_MARGIN_PX = 32.0
"""How far, display px, each element is looked for when it votes on the pose."""

VOTE_MIN_ZNCC = 0.5
"""A match weaker than this is not a vote."""

ROUGH_PX = 3.0
"""How closely, display px, the elements must agree with a rough pose
(:func:`rough_correction`)."""

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
    split: dict | None = None
    """Each element's label (see _explained), for refining the mapping
    without choosing again."""
    model: str = "h"
    """"h" for a homography, "a" for an affine map."""


def can_anchor(profile) -> bool:
    """Whether ``profile`` has enough elements that shift to anchor a pose on."""
    return sum(1 for s in profile if s.kind in SHIFTING) >= MIN_AFFINE


def element_correction(report: RunReport, profile, values=None,
                       home: np.ndarray | None = None) -> Correction | None:
    """The smooth mapping that explains where the elements were found, or None.

    ``home`` as for :func:`far_correction`.
    """
    specs = {s.id: s for s in profile}
    src, dst, ids = [], [], []
    for r in report.results:
        m = r.measurement
        spec = specs.get(r.element_id)
        if (spec is None or spec.kind not in SHIFTING or m.dx is None or m.dy is None
                or m.element_absent or r.reason == "missing_or_displaced"):
            continue
        x, y, w, h = spec.expected_bbox(resolve_value(spec, values))
        src.append((x + w / 2, y + h / 2))
        dst.append((x + w / 2 + m.dx, y + h / 2 + m.dy))
        ids.append(spec.id)
    return _correction(np.float64(src).reshape(-1, 2), np.float64(dst).reshape(-1, 2), ids,
                       home=home)


def far_correction(reference: np.ndarray, live: np.ndarray, profile,
                   values=None, prior: Correction | None = None,
                   home: np.ndarray | None = None) -> Correction | None:
    """The mapping, from every element that shifts looked for up to VOTE_MARGIN_PX.

    Each is a plain ZNCC match of its patch in the reference, both frames in
    display space under the same first pose. Only a vote: what is reported is
    measured afterwards, as usual, under the corrected pose.

    ``home`` is the pose the frames were rectified with, relative to the
    reference's (display to display): where the screen was when it was taught.
    Two explanations can fit equally well -- a speed band of 11 elements moved
    4 px one way, or the view stretched 0.3% and the 4 elements beside it
    moved the other way -- and the one that leaves the screen where it was is
    then taken. ``prior`` is the previous round's correction: the fit starts from its
    split -- which elements the mapping was fitted to and which moved
    together -- and keeps its kind of mapping, rather than choosing again: freshly chosen under each correction, a
    near tie could swing a later round back to a bent mapping. Held fixed
    instead, it kept the rough first round's mistakes, and unchanged screens
    went from 99.1% to 94.9% right; so elements may still change sides as the
    fit settles.
    """
    src, dst, ids = far_votes(reference, live, profile, values)
    return _correction(src, dst, ids, prior, home)


def far_votes(reference: np.ndarray, live: np.ndarray, profile, values=None):
    """Where each element that shifts was found, looked for up to VOTE_MARGIN_PX.

    (where it belongs, where it was found, ids): centres in display px.
    """
    from layoutval.measure import crop, is_degenerate, subpixel_crop, zncc_match

    src, dst, ids = [], [], []
    m = VOTE_MARGIN_PX
    for spec in profile:
        if spec.kind not in SHIFTING:
            continue
        x, y, w, h = spec.expected_bbox(resolve_value(spec, values))
        template = subpixel_crop(reference, (x, y, w, h))
        if template.size == 0 or min(template.shape[:2]) < 3 or is_degenerate(template):
            continue
        sx, sy = int(round(x - m)), int(round(y - m))
        area = crop(live, (sx, sy, int(round(w + 2 * m)), int(round(h + 2 * m))))
        if area.shape[0] < template.shape[0] + 2 or area.shape[1] < template.shape[1] + 2:
            continue
        # Clipped at the frame's edge, the window starts later than asked.
        sx, sy = max(sx, 0), max(sy, 0)
        (px, py), score, on_border = zncc_match(area, template)
        if on_border or score < VOTE_MIN_ZNCC:
            continue
        src.append((x + w / 2, y + h / 2))
        dst.append((sx + px + w / 2, sy + py + h / 2))
        ids.append(spec.id)
    return np.float64(src).reshape(-1, 2), np.float64(dst).reshape(-1, 2), ids


def rough_correction(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, int] | None:
    """A first pose too far out for :func:`far_correction`, brought closer: (G, agreeing).

    One homography that most of the elements agree with to ROUGH_PX, refitted to
    those. Not a measurement: it only brings each element near enough to be
    matched precisely, and far_correction decides afterwards what moved. None
    when no majority agrees.
    """
    n = len(src)
    if n < MIN_AFFINE:
        return None
    G, mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                 cv2.RANSAC, ROUGH_PX, maxIters=4000, confidence=0.999)
    if G is None or mask is None:
        return None
    keep = mask.ravel() > 0
    if keep.sum() < max(MIN_AFFINE, 0.5 * n):
        return None
    refit, _ = cv2.findHomography(src[keep].reshape(-1, 1, 2), dst[keep].reshape(-1, 1, 2), 0)
    if refit is not None:
        G = refit
    G = G / G[2, 2]
    off = np.linalg.norm(_project(G[None], src)[0] - dst, axis=1)
    return G, int((off < ROUGH_PX).sum())


def _correction(src_a: np.ndarray, dst_a: np.ndarray, ids: list[str],
                prior: Correction | None = None,
                home: np.ndarray | None = None) -> Correction | None:
    """The mapping from where the elements belong to where they were found."""
    n = len(src_a)
    if n < MIN_AFFINE:
        return None
    if prior is not None and prior.split is not None:
        label = np.array([prior.split.get(i, UNEXPLAINED) for i in ids])
        model = prior.model
        found = _refine(src_a, dst_a, label, model)
    else:
        found = _choose(src_a, dst_a, home)
        if found is None:
            return None
        G, label, model = found
        found = (G, label)
    if found is None:
        return None
    G, label = found
    inliers, grouped = int((label == POSE).sum()), int((label > POSE).sum())
    if inliers + grouped < max(MIN_AFFINE - 2, 0.6 * n) or inliers < MIN_AFFINE - 2:
        return None                       # no one mapping explains most of them
    moved = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), G).reshape(-1, 2) - src_a
    return Correction(G=G, used=n, inliers=inliers, grouped=grouped,
                      largest_px=float(np.linalg.norm(moved, axis=1).max()),
                      split=dict(zip(ids, label.tolist())), model=model)


UNEXPLAINED, POSE = -1, 0
"""Labels: neither in the mapping nor in a group; in the mapping. A group is 1, 2, ..."""


def _project(Hs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points through each of a stack of homographies: (k, n, 2)."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    q = np.einsum("kij,nj->kni", Hs, p)
    return q[..., :2] / q[..., 2:3]


def _explained(residuals: np.ndarray) -> np.ndarray:
    """Each element's label: in the mapping, in a group that moved together, or neither.

    Groups are taken largest first, each the elements whose shifts lie within
    INLIER_PX of one of them: a fault can move more than one thing, and by
    different amounts -- a gauge 4 px and a gear letter 3 px.
    """
    norm = np.linalg.norm(residuals, axis=1)
    label = np.full(len(residuals), UNEXPLAINED)
    label[norm < INLIER_PX] = POSE
    rest = np.flatnonzero((label == UNEXPLAINED) & (norm >= GROUP_SHIFT_PX))
    group = POSE
    while len(rest) >= GROUP_MIN and group < MAX_GROUPS:
        r = residuals[rest]
        together = np.linalg.norm(r[:, None] - r[None], axis=2) < INLIER_PX
        centre = int(together.sum(axis=1).argmax())
        if together[centre].sum() < GROUP_MIN:
            break
        group += 1
        label[rest[together[centre]]] = group
        rest = rest[~together[centre]]
    return label


def _score(label: np.ndarray) -> tuple[int, int] | None:
    """(explained, in the mapping), or None if a group outnumbers the mapping.

    The mapping is the camera, and the camera is what most of the screen
    agrees on. Without that, a mapping lined up on what moved scored as well
    as the right one -- the moved elements its inliers, the rest one big group
    -- and with two groups moved by different amounts it scored better.
    """
    inliers = int((label == POSE).sum())
    sizes = np.bincount(label[label > POSE]) if (label > POSE).any() else np.zeros(1, int)
    if sizes.max() > inliers:
        return None
    return inliers + int(sizes.sum()), inliers


def _choose(src: np.ndarray, dst: np.ndarray, home: np.ndarray | None = None):
    """RANSAC for which elements the mapping is fitted to and which moved together.

    Returns (G, label, model) of the explanation taken, or None.
    """
    n = len(src)
    homography = n >= MIN_HOMOGRAPHY
    rng = np.random.default_rng(0)
    k = 4 if homography else 3
    samples = np.array([rng.choice(n, k, replace=False) for _ in range(HYPOTHESES)])
    stack = np.full((HYPOTHESES, 3, 3), np.nan)
    for i, s in enumerate(samples):
        a, b = src[s].astype(np.float32), dst[s].astype(np.float32)
        try:
            stack[i] = (cv2.getPerspectiveTransform(a, b) if homography
                        else np.vstack([cv2.getAffineTransform(a, b), [0.0, 0.0, 1.0]]))
        except cv2.error:
            pass                          # points in a line
    # And every element's own shift, as the whole screen's. A shift cannot
    # bend, so the plain answer -- the screen where it was, one group moved --
    # is always among the candidates. Drawn from four elements at a time, it
    # need not be: on a hazy webcam shot of a 32-element cluster with its speed
    # band moved, the four-element mappings came out bent -- stretched 0.3% to
    # follow the band, the screen's middle 1.7 px off -- and one of those won,
    # passing the band and failing the rest, on one capture in nine.
    shifts = np.repeat(np.eye(3)[None], n, axis=0)
    shifts[:, :2, 2] = dst - src
    stack = np.concatenate([stack, shifts])
    # And the screen exactly where it was taught. A first pose that ECC let a
    # moved band pull was 1-2% stretched and 6 px off, and most of what had
    # not moved stood in one column down the right: no four of them drawn at
    # random gave back the plain answer, which this is.
    taught = None
    if home is not None:
        taught = len(stack)
        stack = np.concatenate([stack, np.linalg.inv(home)[None]])
    with np.errstate(all="ignore"):
        residuals = _project(stack, src) - dst[None]
    finite = np.isfinite(residuals).all(axis=(1, 2))
    norms = np.linalg.norm(np.where(finite[:, None, None], residuals, np.inf), axis=2)
    counts = (norms < INLIER_PX).sum(axis=1)
    # The most a hypothesis can explain is its inliers and everything far
    # enough off to be in a group; tried best first, the rest are skipped once
    # they cannot win.
    bound = counts + (np.isfinite(norms) & (norms >= GROUP_SHIFT_PX)).sum(axis=1)
    order = np.lexsort((-counts, -bound))
    # Each hypothesis as drawn is a rough start: from four noisy elements it
    # extrapolates badly. The most promising are refitted to what they agree
    # with before they are compared (locally optimised RANSAC). Compared as
    # drawn, on a hazy webcam shot at 0.51 camera px per display px -- each
    # element placed to 0.5-1 px -- the plain answer lost its supporters to
    # the noise, and a mapping bent to follow a moved speed band won.
    # The best REFINED distinct labellings; a hypothesis's key can be no more
    # than (bound, count), so once that falls below the weakest kept, the rest
    # are skipped.
    top: dict[bytes, tuple[tuple[int, int], np.ndarray, int]] = {}
    if taught is not None and finite[taught]:
        label = _explained(residuals[taught])
        key = _score(label)
        if key is not None:
            top[label.tobytes()] = (key, label, taught)
    for i in order:
        if not finite[i] or counts[i] < max(3, 0.2 * n):
            continue
        if len(top) >= REFINED and (bound[i], counts[i]) < min(v[0] for v in top.values()):
            break
        label = _explained(residuals[i])
        key = _score(label)
        if key is None:
            continue
        signature = label.tobytes()
        if signature in top and top[signature][0] >= key:
            continue
        top[signature] = (key, label, i)
        if len(top) > REFINED:
            del top[min(top, key=lambda s: top[s][0])]
    done = []
    for key, label, i in sorted(top.values(), key=lambda v: v[0], reverse=True):
        # Refitted, or kept as drawn where the refit fails or does worse: the
        # taught pose, refitted as an affine map from the 16 elements it held,
        # could not undo a first pose bent by perspective, and was lost.
        model = "h" if i == taught or (label == POSE).sum() >= MIN_HOMOGRAPHY else "a"
        refined = _refine(src, dst, label, model, start=stack[i])
        refined_key = None if refined is None else _score(refined[1])
        if refined_key is not None and refined_key >= key:
            done.append((refined_key, refined[0], refined[1], model))
        else:
            done.append((key, stack[i], label, "h" if homography or i == taught else "a"))
    if not done:
        return None
    most = max(d[0][0] for d in done)
    close = [d for d in done if d[0][0] >= most - TIE_ELEMENTS]
    # Among those that explain about as much as the best, the one that leaves
    # the screen nearest where it was taught -- but only if the screen where it
    # was taught is itself among them. When the camera or the lid has moved,
    # it explains little, and which of the rest is nearer means nothing.
    if home is not None and any(_away(home, d[1], src) < AT_HOME_PX for d in close):
        pick = min(close, key=lambda d: (_away(home, d[1], src), -d[0][1]))
    else:
        pick = max(done, key=lambda d: d[0])
    return pick[1], pick[2], pick[3]


def _away(home: np.ndarray, G: np.ndarray, pts: np.ndarray) -> float:
    """How far, display px on average, pose ``G`` puts the elements from where they were taught."""
    there = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), home @ G).reshape(-1, 2)
    return float(np.linalg.norm(there - pts, axis=1).mean())


def _fit(model: str, src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least squares: a homography ("h") or an affine map ("a"), as 3x3."""
    if model == "h":
        if len(src) < 4:
            return None
        G, _ = cv2.findHomography(src, dst, 0)
        return G
    if len(src) < 3:
        return None
    A = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    X, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return np.vstack([X.T, [0.0, 0.0, 1.0]])


def _refine(src: np.ndarray, dst: np.ndarray, label: np.ndarray, model: str = "h",
            start: np.ndarray | None = None):
    """Fit the mapping to its elements with each group as one block with its own shift.

    A group's shape pins the side of the screen it is on, where nothing else
    may. Which elements are which is updated as the fit settles, and the
    best-scoring state is what is returned: with few elements, an update could
    drop a handful out of the mapping, which then fitted the rest worse and
    dropped more -- 21 elements in it, then 14, then 11, and the pose 11 px
    out on one capture of a scene the others got right. Returns (G, label),
    or None.
    """
    G = start if start is not None else _fit(model, src[label == POSE], dst[label == POSE])
    if G is None:
        return None
    best, best_key = None, None
    # residual = mapped - found, so a group's own shift is minus its residual
    shifts = {}
    for _ in range(20):
        residual = _project(G[None], src)[0] - dst
        new = {g: -np.median(residual[label == g], axis=0)
               for g in np.unique(label[label > POSE])}
        fit = label >= POSE
        moved_dst = dst.copy()
        for g, s in new.items():
            moved_dst[label == g] -= s
        G = _fit(model, src[fit], moved_dst[fit])
        if G is None:
            return None
        residual = _project(G[None], src)[0] - dst
        now = _explained(residual)
        key = _score(now)
        # A later state that scores as well is the more settled one.
        if key is not None and (best_key is None or key >= best_key):
            best, best_key = (G.copy(), now.copy()), key
        if (now == POSE).sum() < 3:
            break
        settled = ((now == label).all() and new.keys() == shifts.keys()
                   and all(np.abs(new[g] - shifts[g]).max() < 0.01 for g in new))
        label, shifts = now, new
        if settled:
            break
    return best

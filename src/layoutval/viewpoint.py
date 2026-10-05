"""How far the camera's view moved since the reference, and what that alone can shift.

A homography maps a flat screen exactly, and no screen is quite flat. On six
photographs of a laptop's cluster, from in front, from well above, below and
either side, with nothing on it changed, the best homography -- found from the
screen's own corners, then fitted to all 35 elements -- left the middle of the
screen 3-3.5 px out against its sides -- in four of five pairs along the way
the camera had moved, and by as much as that predicts: 3.5 px left when it
moved left, 3 px down when it moved down. That is what a screen about a
millimetre out of flat shows from that far round (parallax: a point off the
plane slides along the camera's move, by its distance off the plane times how
far the view turned). From a steep view, the same millimetre is 5-9 px, and
the direction is only roughly known. No pose removes it, and from one
photograph each it cannot be told from an element drawn that much out of
place in that direction.

So the view's move is estimated from the two poses -- each camera's position
over the screen, from its homography and a focal length -- and at each element
the shift a screen FLATNESS out of flat can show is worked out, along with its
direction. An element that fails only by that much, along that direction, is
REVIEW rather than FAIL, and the report says why: retaken from nearer the
reference position, it is decided. Nothing is ever passed this way, and across
it -- or by more -- an element fails as before. Where the unevenness is more
than the failure tolerance, the run says it cannot be judged at that tolerance
from there, and how close to come. With the camera where it was, or moved a
little, the allowance is under MIN_ALLOWANCE_PX and nothing changes.
"""

from __future__ import annotations

import cv2
import numpy as np

from layoutval.types import Verdict, resolve_value
from layoutval.verdict import POSITION

FLATNESS = 0.004
"""How far out of flat a screen may be, as a fraction of its width: 1.2 mm on a
30 cm laptop screen, which covered the laptop's 1 mm."""

MIN_ALLOWANCE_PX = 1.0
"""An allowance smaller than this, display px, changes nothing: the view barely moved."""

FOCAL_OF_DIAGONAL = 0.6
"""Focal length as a fraction of the frame's diagonal, when no lens was solved:
a phone's main camera (26 mm equivalent) is 0.60, a laptop webcam 0.55-0.65."""

VIEW_ANGLE = "position_view_angle"
"""Reason code: off by no more than the change of view can explain."""


def default_camera_matrix(shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    f = FOCAL_OF_DIAGONAL * float(np.hypot(w, h))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def camera_centre(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Where the camera was, in display px: (x, y) over the screen and its distance."""
    M = np.linalg.inv(K) @ H
    scale = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if M[2, 2] * scale < 0:
        scale = -scale                     # the screen is in front of the camera
    r1, r2, t = scale * M[:, 0], scale * M[:, 1], scale * M[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    C = -R.T @ t
    return np.array([C[0], C[1], abs(C[2])])


def parallax(H_ref: np.ndarray, H_live: np.ndarray, K: np.ndarray,
             pts: np.ndarray) -> np.ndarray:
    """Shift at each display point, display px, per display px it lies off the plane."""
    a, b = camera_centre(H_ref, K), camera_centre(H_live, K)
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return (b[:2] - pts) / b[2] - (a[:2] - pts) / a[2]


def view_angle_deg(H_ref: np.ndarray, H_live: np.ndarray, K: np.ndarray,
                   at: tuple[float, float]) -> float:
    """Angle between the two lines of sight to the display point ``at``."""
    p = np.array([at[0], at[1], 0.0])
    u, v = camera_centre(H_ref, K) - p, camera_centre(H_live, K) - p
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


SOFTEN_SIGMA = 1.5
"""Blur, display px, both patches get before identity is checked again from afar."""


def _allowance(H_ref, H_live, K, spec, values, display_size):
    """(allowance px, unit axis) at ``spec``'s centre."""
    x, y, w, h = spec.expected_bbox(resolve_value(spec, values))
    e = parallax(H_ref, H_live, K, [(x + w / 2, y + h / 2)])[0]
    n = float(np.linalg.norm(e))
    return FLATNESS * display_size[0] * n, (e / n if n > 0 else e)


def recheck_from_afar(report, profile, values, reference: np.ndarray, live: np.ndarray,
                      H_ref: np.ndarray, H_live: np.ndarray, K: np.ndarray,
                      display_size: tuple[int, int]) -> list[str]:
    """Check identity again, softened, where the view moved and it failed. Returns the ids changed.

    Seen from far round, the same label looks different: on the laptop's
    photographs the one from close in showed thin strokes on a hazy screen,
    the one from further off the same strokes bold, bloomed and edged with the
    phone's sharpening -- correlation 0.69-0.77 for the same "50", under the
    0.80 that says it is the same symbol. Blurred SOFTEN_SIGMA px, both are the
    same shape again: 16 of 24 such failures matched, while of 399 pairs of
    different elements from the same cluster only one more reached 0.80 (from
    0.78) -- the pairs that already did were labels drawn alike, a "0" and a
    "0". Only where the view moved enough to matter (MIN_ALLOWANCE_PX), so a
    camera that stayed put is judged exactly as before.
    """
    from layoutval.capture import to_gray
    from layoutval.measure import subpixel_crop
    from layoutval.verdict import WRONG_CONTENT, evaluate

    specs = {s.id: s for s in profile}
    changed = []
    for i, r in enumerate(report.results):
        m, spec = r.measurement, specs.get(r.element_id)
        if r.reason != WRONG_CONTENT or spec is None or m.dx is None or m.dy is None:
            continue
        if _allowance(H_ref, H_live, K, spec, values, display_size)[0] < MIN_ALLOWANCE_PX:
            continue
        x, y, w, h = spec.expected_bbox(resolve_value(spec, values))
        a = to_gray(subpixel_crop(reference, (x, y, w, h))).astype(np.float64)
        b = to_gray(subpixel_crop(live, (x + m.dx, y + m.dy, w, h))).astype(np.float64)
        if a.shape != b.shape or min(a.shape) < 3:
            continue
        a = cv2.GaussianBlur(a, (0, 0), SOFTEN_SIGMA)
        b = cv2.GaussianBlur(b, (0, 0), SOFTEN_SIGMA)
        a, b = a - a.mean(), b - b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        score = float((a * b).sum()) / den if den > 0 else 0.0
        if score >= r.tolerance.identity_min:
            m.zncc = score
            m.method += " (softened: seen from afar)"
            report.results[i] = evaluate(spec, m)
            changed.append(spec.id)
    return changed


def allow_for_view(report, profile, values, H_ref: np.ndarray, H_live: np.ndarray,
                   K: np.ndarray, display_size: tuple[int, int],
                   tol_fail: float | None = None) -> int:
    """Turn position failures the change of view explains into REVIEW. Returns how many.

    And say so when the view moved so far that a screen's own unevenness, at
    its centre, is more than ``tol_fail``: past that a fault the size of the
    tolerance cannot be told from it. On the laptop's photographs, 13-14
    degrees round (2.0-2.1 px) a 4 px move of the speed band was still found
    whole; 25 degrees round (3.5-3.8 px), the same move was found as REVIEW at
    best, and once not at all.
    """
    specs = {s.id: s for s in profile}
    dw, dh = display_size
    centre = FLATNESS * dw * float(np.linalg.norm(
        parallax(H_ref, H_live, K, [(dw / 2, dh / 2)])[0]))
    # Past the tolerance, the run is not judged at it, and the direction an
    # uneven screen shifts in is itself only roughly known: from 43 degrees
    # round, the middle of the laptop's screen was 3.3-3.7 px out, partly
    # across the camera's move. Then any position failure no larger than the
    # unevenness is REVIEW; a larger one still fails.
    too_far = tol_fail is not None and centre > tol_fail
    changed, largest = [], 0.0
    for r in report.results:
        m, spec = r.measurement, specs.get(r.element_id)
        if (r.verdict is not Verdict.FAIL or r.reason != POSITION or spec is None
                or m.dx is None or m.dy is None):
            continue
        allowance, axis = _allowance(H_ref, H_live, K, spec, values, display_size)
        if allowance < MIN_ALLOWANCE_PX:
            continue
        d = np.array([m.dx, m.dy])
        along = float(d @ axis)
        rest = d - along * axis + np.sign(along) * max(0.0, abs(along) - allowance) * axis
        if (float(np.linalg.norm(rest)) <= r.tolerance.tol_fail
                or (too_far and float(np.linalg.norm(d)) <= allowance)):
            r.verdict, r.reason = Verdict.REVIEW, VIEW_ANGLE
            changed.append(r.element_id)
            largest = max(largest, allowance)
    if not changed and not too_far:
        return 0
    angle = view_angle_deg(H_ref, H_live, K, (dw / 2, dh / 2))
    parts = [f"the camera looks at the screen from about {angle:.0f} degrees away from "
             "where the reference was taken. No screen is perfectly flat -- a laptop's lid "
             "is out by a millimetre or so -- and from that far round that alone shifts "
             f"parts of it by up to {max(largest, centre):.1f} px, along the way the camera "
             "moved."]
    if changed:
        parts.append(f"{len(changed)} element(s) off by no more than that are REVIEW, not "
                     "FAIL.")
    if too_far:
        within = max(1.0, angle * tol_fail / centre)
        parts.append(f"That is more than the {tol_fail:.1f} px a failure is called at, so a "
                     "fault that small cannot be told from it here, and what passed is not "
                     f"proven. Take the photo within about {within:.0f} degrees of where the "
                     "reference was taken.")
    else:
        parts.append("Take the photo from nearer where the reference was taken to decide "
                     "them.")
    report.flag("view_moved", severity="review", elements=changed,
                angle_deg=round(angle, 1), allowance_px=round(max(largest, centre), 1),
                too_far=too_far, detail=" ".join(parts))
    return len(changed)

"""Find the display with SAM 3, served from another computer.

SAM 3 (Meta's Segment Anything Model 3) segments whatever a short text prompt
names -- "display screen", "round instrument cluster" -- and it does not care
what shape the thing is. That is what the corner routes cannot do: they look
for four straight edges, and a round cluster has none. So the photograph is
sent, with the prompt, to a SAM 3 service on a computer with a GPU
(``sam3_server/sam3_server.py``), the mask comes back, and the display's four
framebuffer corners are worked out from its outline:

* **Four straight sides** (a mask its fitted quadrilateral covers to
  QUAD_FIT): each side is a line fitted to the outline between the corners,
  away from them, so rounded corners do not pull it; the corners are where the
  lines meet. They are then snapped to the panel's real edge, as tapped
  corners are.
* **Round or oval** (its fitted ellipse covers it to ELLIPSE_FIT): the
  framebuffer is the rectangle the round screen is inscribed in, mapped by the
  affine map taking that ellipse in display space to the one in the
  photograph. A circle has no corners to say which way up it is, so the
  camera is taken to be upright; drag the dots to turn it.
* **Anything else**: the smallest rotated rectangle round the outline, not
  confident.

The corners go to the phone as dots to check, exactly like the automatic
corner proposal, and nothing is calibrated until they are confirmed: a learned
model proposes, a person accepts, and the measurement never runs one
(docs/architecture.md, section 3). This module only talks HTTP; it imports no
model and needs nothing beyond what layoutval already uses.

The outline also says which part of display space is screen: for a round
cluster, the corners of its framebuffer are bezel, and the inventory and the
comparisons stay inside the circle (:func:`display_mask`).
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_PROMPT = "display screen"
"""What SAM 3 is asked for when nothing else is typed."""

TIMEOUT_S = 120.0
"""How long to wait for the SAM 3 computer: the first request after it starts
also loads the model."""

SEND_SIDE = 2048
"""Longest side of the photograph as sent. SAM 3 works at 1008 px anyway, so a
phone's full 12-24 MP only costs time on the network."""

THRESHOLD = 0.4
"""Least SAM 3 score for a mask to come back."""

QUAD_FIT = 0.97
"""Least overlap (IoU) of the mask with its fitted quadrilateral to call the
display four-sided."""

ELLIPSE_FIT = 0.97
"""Least overlap (IoU) of the mask with its fitted ellipse to call it round."""

CONFIDENT_FIT = 0.985
"""Overlap at which a shape is shown as found rather than to be checked."""


class Sam3Error(RuntimeError):
    """The SAM 3 computer could not be reached, or refused, or answered nonsense."""


@dataclass
class Segment:
    mask: np.ndarray
    """Boolean, the photograph's own size."""
    score: float
    box: tuple[float, float, float, float]
    """x0, y0, x1, y1 in the photograph's pixels."""


@dataclass
class DisplayShape:
    kind: str
    """``rect``, ``round`` or ``other``."""
    corners: np.ndarray
    """The framebuffer's corners in the photograph: top-left, top-right,
    bottom-right, bottom-left, (4, 2)."""
    outline: np.ndarray
    """The screen's outline in the photograph, (n, 2), simplified."""
    fit: float
    """Overlap of the mask with the shape fitted to it, 0..1."""
    confident: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self, shape: tuple[int, ...]) -> dict:
        """For the phone: corners and outline as fractions of the photograph."""
        h, w = shape[:2]
        frac = lambda pts: [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in pts]
        return {"corners": frac(self.corners), "confident": self.confident,
                "shape": self.kind, "fit": round(self.fit, 3), "outline": frac(self.outline)}


# --------------------------------------------------------------------------
# the SAM 3 computer
# --------------------------------------------------------------------------


def segment(image: np.ndarray, url: str, prompt: str, *, token: str | None = None,
            threshold: float = THRESHOLD, limit: int = 5,
            timeout: float = TIMEOUT_S) -> list[Segment]:
    """Masks SAM 3 finds for ``prompt`` in ``image`` (BGR), best first.

    Raises :class:`Sam3Error` when the service cannot be reached or refuses.
    """
    h, w = image.shape[:2]
    scale = min(1.0, SEND_SIDE / max(h, w))
    sent = image if scale == 1.0 else cv2.resize(
        image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", sent, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise Sam3Error("could not encode the photograph")
    query = urllib.parse.urlencode({"prompt": prompt, "threshold": threshold, "max": limit})
    request = urllib.request.Request(
        url.rstrip("/") + "/segment?" + query, data=jpeg.tobytes(), method="POST",
        headers={"Content-Type": "image/jpeg", **({"X-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            reason = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:  # noqa: BLE001
            reason = ""
        if exc.code == 403:
            raise Sam3Error("the SAM 3 computer refused the token -- start layoutval "
                            "with the same --sam3-token the server has") from None
        raise Sam3Error(f"the SAM 3 computer answered {exc.code}"
                        + (f": {reason}" if reason else "")) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise Sam3Error(f"could not reach the SAM 3 computer at {url} ({reason})") from None

    sw, sh = sent.shape[1], sent.shape[0]
    found = []
    for r in payload.get("results", []):
        raw = cv2.imdecode(np.frombuffer(base64.b64decode(r["mask_png"]), np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (sh, sw):
            continue
        mask = raw if scale == 1.0 else cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        box = tuple(float(v) / scale for v in r.get("box", (0, 0, 0, 0)))
        found.append(Segment(mask=mask >= 128, score=float(r.get("score", 0.0)), box=box))
    return found


def health(url: str, *, timeout: float = 5.0) -> dict:
    """The service's own report, or :class:`Sam3Error`."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise Sam3Error(f"could not reach the SAM 3 computer at {url} "
                        f"({getattr(exc, 'reason', exc)})") from None


def best(segments: list[Segment], near: np.ndarray | None = None) -> Segment | None:
    """The mask that is the display: the one most over ``near`` (a boolean
    region where it should be), else the best scored."""
    if not segments:
        return None
    if near is not None:
        def overlap(s: Segment) -> float:
            inter = float(np.logical_and(s.mask, near).sum())
            union = float(np.logical_or(s.mask, near).sum())
            return inter / union if union else 0.0
        scored = max(segments, key=overlap)
        if overlap(scored) > 0.3:
            return scored
    return max(segments, key=lambda s: (s.score, int(s.mask.sum())))


# --------------------------------------------------------------------------
# the shape of the mask
# --------------------------------------------------------------------------


def _clean(mask: np.ndarray) -> np.ndarray:
    """The largest piece of the mask, holes filled, as uint8 0/1."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == keep).astype(np.uint8)
    # Content drawn on the screen can leave holes in a mask of it.
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _order(corners: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    c = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    corners = corners[np.argsort(ang)]                  # clockwise in image coords
    start = int(np.argmin(corners.sum(axis=1)))         # top-left: smallest x + y
    return np.roll(corners, -start, axis=0)


def _quad(contour: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Four corners from the outline's four straight sides, and their fit."""
    hull = cv2.convexHull(contour.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            approx = a.reshape(4, 2).astype(np.float64)
            break
    if approx is None:
        return None
    approx = _order(approx)
    pts = contour.reshape(-1, 2).astype(np.float64)
    lines = []
    for k in range(4):
        p, q = approx[k], approx[(k + 1) % 4]
        d = q - p
        length = float(np.hypot(*d))
        if length < 10:
            return None
        u = d / length
        rel = pts - p
        t = rel @ u
        off = np.abs(rel @ np.array([-u[1], u[0]]))
        # The middle of the side only: the corners may be rounded.
        on = (t > 0.12 * length) & (t < 0.88 * length) & (off < max(3.0, 0.03 * length))
        if on.sum() < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts[on].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    corners = []
    for k in range(4):
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        a = np.array([d1, -d2]).T
        if abs(np.linalg.det(a)) < 1e-9:
            return None
        s, _ = np.linalg.solve(a, p2 - p1)
        corners.append(p1 + s * d1)
    corners = np.array(corners)
    filled = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(filled, np.round(corners).astype(np.int32), 1)
    return corners, filled


def _ellipse(contour: np.ndarray, shape: tuple[int, int]):
    if len(contour) < 20:
        return None
    (cx, cy), (aw, ah), angle = cv2.fitEllipse(contour.astype(np.float32))
    if min(aw, ah) < 10:
        return None
    filled = np.zeros(shape, np.uint8)
    cv2.ellipse(filled, ((cx, cy), (aw, ah), angle), 1, thickness=cv2.FILLED)
    return (cx, cy, aw / 2.0, ah / 2.0, np.radians(angle)), filled


def fit_display(mask: np.ndarray, display_size: tuple[int, int]) -> DisplayShape | None:
    """The display's framebuffer corners from a SAM 3 mask of the screen.

    ``display_size`` is the framebuffer's (width, height): for a round screen,
    the rectangle it is inscribed in.
    """
    m = _clean(mask)
    if m.sum() < 400:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    shape = m.shape[:2]
    on = m > 0
    peri = cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.float32), True)
    outline = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.float32),
                               0.002 * peri, True).reshape(-1, 2)

    quad = _quad(contour, shape)
    quad_fit = _iou(on, quad[1] > 0) if quad else 0.0
    ell = _ellipse(contour, shape)
    ell_fit = _iou(on, ell[1] > 0) if ell else 0.0

    if quad and quad_fit >= QUAD_FIT and quad_fit >= ell_fit:
        return DisplayShape("rect", _order(quad[0]), outline, quad_fit,
                            quad_fit >= CONFIDENT_FIT)
    if ell and ell_fit >= ELLIPSE_FIT:
        cx, cy, a, b, theta = ell[0]
        # The affine map with no turn of its own (symmetric) taking the unit
        # circle to this ellipse: a round screen gives no clue which way is up.
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        A = rot @ np.diag([a, b]) @ rot.T
        unit = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        corners = unit @ A.T + np.array([cx, cy])
        return DisplayShape("round", corners, outline, ell_fit, ell_fit >= CONFIDENT_FIT,
                            extra={"ellipse": [float(cx), float(cy), float(a), float(b),
                                               float(np.degrees(theta))]})
    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    filled = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(filled, np.round(box).astype(np.int32), 1)
    return DisplayShape("other", _order(box.astype(np.float64)), outline,
                        _iou(on, filled > 0), False)


def display_mask(mask: np.ndarray, H: np.ndarray, display_size: tuple[int, int],
                 *, shrink: float = 0.005) -> np.ndarray:
    """The screen's own area in display space: the mask taken through ``H``
    (display -> photograph), less a thin rim where SAM 3's edge is uncertain."""
    dw, dh = display_size
    warped = cv2.warpPerspective(_clean(mask) * 255, np.linalg.inv(H), (dw, dh),
                                 flags=cv2.INTER_LINEAR)
    inside = (warped >= 128).astype(np.uint8)
    k = max(1, int(round(shrink * min(dw, dh))))
    return cv2.erode(inside, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0


def draw(image: np.ndarray, shape: DisplayShape) -> np.ndarray:
    """The photograph with the outline SAM 3 found and the corners from it."""
    out = image.copy()
    t = max(2, round(max(image.shape[:2]) / 600))
    cv2.polylines(out, [np.round(shape.outline).astype(np.int32).reshape(-1, 1, 2)], True,
                  (255, 160, 60), t, cv2.LINE_AA)
    cv2.polylines(out, [np.round(shape.corners).astype(np.int32).reshape(-1, 1, 2)], True,
                  (80, 200, 120), t, cv2.LINE_AA)
    for i, (x, y) in enumerate(shape.corners):
        cv2.circle(out, (int(round(x)), int(round(y))), 4 * t, (80, 200, 120), -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (int(round(x)) + 6 * t, int(round(y)) - 6 * t),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (80, 200, 120), t, cv2.LINE_AA)
    return out

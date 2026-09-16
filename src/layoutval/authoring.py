"""Assisted annotation -- the one honest place for a learned model.

The line worth holding: **learned models in the authoring loop, deterministic
code in the measurement loop.**  It is not an anti-ML position.  It puts ML
where its failure mode is "an engineer adjusts a box" rather than "a safety
telltale defect ships".

Nothing in this module is imported by :mod:`layoutval.pipeline`,
:mod:`layoutval.measure` or :mod:`layoutval.verdict`, and
``tests/test_architecture.py`` enforces that.  A proposal only becomes an
element after :func:`accept_proposal` is called with an explicit human
confirmation.

Snap assistance (:func:`snap_box`, :func:`mser_candidates`) is deterministic and
needs no model at all; reach for a proposer only for the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np

from layoutval.capture import to_gray
from layoutval.types import ElementKind, ElementSpec, PositionModel, Tolerance


class UnconfirmedProposal(RuntimeError):
    """Raised when a model proposal is about to enter the inventory unreviewed."""


@dataclass
class Proposal:
    """A candidate element box awaiting an engineer's confirmation."""

    bbox: tuple[int, int, int, int]
    score: float = 0.0
    origin: str = "snap"
    """``snap``, ``mser`` or the proposer's name (e.g. ``sam2``)."""

    mask: np.ndarray | None = None


class MaskProposer(Protocol):
    """A segmentation backend used **during authoring only**.

    SAM and SAM 2 are Apache-2.0, run offline, and a human approves every output,
    so nothing stochastic reaches the measurement path.  Grounding DINO /
    Grounded-SAM (also Apache-2.0) fit the same slot for text-prompted boxes.

    Deliberately *not* a dependency of this package: the implementation lives in
    whatever authoring environment you run, and this protocol is all the
    measurement code ever needs to know about it.
    """

    def propose(
        self, image: np.ndarray, point: tuple[int, int]
    ) -> np.ndarray:  # pragma: no cover - protocol
        """Return a boolean or 0/255 mask for the object under ``point``."""


def snap_box(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    window: int = 96,
    threshold: int | None = None,
) -> Proposal | None:
    """Deterministic snap assistance: grow the connected component under a click.

    Good enough for most lit elements on a dark cluster background, with no model
    involved.  Try this before reaching for a proposer.
    """
    gray = to_gray(image)
    h, w = gray.shape[:2]
    px, py = int(point[0]), int(point[1])
    x0, y0 = max(0, px - window), max(0, py - window)
    x1, y1 = min(w, px + window), min(h, py + window)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    if threshold is None:
        _, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(patch, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    label = int(labels[py - y0, px - x0])
    if label == 0:
        # Clicked on background: fall back to the nearest component in the window.
        if n <= 1:
            return None
        label = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, bw, bh = (int(stats[label, k]) for k in range(4))
    return Proposal(
        bbox=(x0 + x, y0 + y, bw, bh),
        score=float(stats[label, cv2.CC_STAT_AREA]),
        origin="snap",
        mask=(labels == label).astype(np.uint8) * 255,
    )


def mser_candidates(
    image: np.ndarray, *, min_area: int = 40, max_area: int = 20000, delta: int = 5
) -> list[Proposal]:
    """MSER regions as snap targets for an authoring UI.

    Useful for text runs and small icons, where a single threshold does not
    separate the element from a gradient background.  It needs intensity
    structure to work with: on flat synthetic shapes it legitimately returns
    nothing, and :func:`snap_box` is the right tool there.
    """
    mser = cv2.MSER.create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)
    regions, _ = mser.detectRegions(to_gray(image))
    out: list[Proposal] = []
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        out.append(Proposal(bbox=(x, y, w, h), score=float(len(pts)), origin="mser"))
    out.sort(key=lambda p: p.score, reverse=True)
    return out


def propose_with_model(
    proposer: MaskProposer, image: np.ndarray, point: tuple[int, int], *, name: str = "model"
) -> Proposal | None:
    """Ask a segmentation model for a mask and tighten it to a box.

    Authoring only.  The engineer confirms every output before
    :func:`accept_proposal` will turn it into an element.
    """
    mask = proposer.propose(image, point)
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    return Proposal(bbox=(x, y, w, h), score=float(mask.sum()) / 255.0, origin=name, mask=mask)


def accept_proposal(
    proposal: Proposal,
    element_id: str,
    *,
    confirmed_by: str,
    kind: ElementKind = ElementKind.ICON,
    tolerance: Tolerance | None = None,
    notes: str = "",
) -> ElementSpec:
    """Turn a confirmed proposal into an inventory element.

    ``confirmed_by`` is required and must be a real identifier.  It is written
    into the element's notes so the provenance of a manually authored expectation
    survives into the report -- which is the whole point of keeping the model out
    of the measurement loop.
    """
    if not confirmed_by or not confirmed_by.strip():
        raise UnconfirmedProposal(
            f"{element_id}: proposals from {proposal.origin!r} need an explicit "
            "human confirmation before they enter the inventory"
        )
    x, y, w, h = proposal.bbox
    trail = f"proposed by {proposal.origin}, confirmed by {confirmed_by}"
    return ElementSpec(
        id=element_id,
        kind=kind,
        bbox=(float(x), float(y), float(w), float(h)),
        position=PositionModel(origin=(float(x), float(y))),
        tolerance=tolerance or Tolerance(),
        source="manual",
        notes=f"{trail}. {notes}".strip(),
    )


def deduplicate(proposals: Sequence[Proposal], *, iou_threshold: float = 0.6) -> list[Proposal]:
    """Drop overlapping candidates, keeping the highest-scoring one.

    An authoring UI that shows forty overlapping MSER boxes over one icon is one
    an engineer stops using.
    """

    def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    kept: list[Proposal] = []
    for p in sorted(proposals, key=lambda q: q.score, reverse=True):
        if all(iou(p.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(p)
    return kept

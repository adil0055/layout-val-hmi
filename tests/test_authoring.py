"""The authoring loop: assistance is welcome, unconfirmed proposals are not."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval.authoring import (
    Proposal,
    UnconfirmedProposal,
    accept_proposal,
    deduplicate,
    mser_candidates,
    propose_with_model,
    snap_box,
)
from layoutval.types import ElementKind


def _scene() -> np.ndarray:
    img = np.full((120, 160, 3), 8, np.uint8)
    cv2.rectangle(img, (40, 30), (75, 60), (220, 220, 220), -1)
    cv2.circle(img, (120, 80), 12, (200, 200, 200), -1)
    return img


def test_snap_box_grows_the_component_under_the_click():
    p = snap_box(_scene(), (55, 45))
    assert p is not None
    x, y, w, h = p.bbox
    assert abs(x - 40) <= 2 and abs(y - 30) <= 2
    assert abs(w - 36) <= 3 and abs(h - 31) <= 3
    assert p.origin == "snap"


def test_snap_box_falls_back_to_the_nearest_component_on_a_background_click():
    p = snap_box(_scene(), (30, 20))
    assert p is not None and p.bbox[2] > 0


def test_snap_box_returns_none_outside_the_image():
    assert snap_box(_scene(), (500, 500)) is None


def test_mser_offers_candidates_on_content_with_structure():
    """MSER needs intensity structure; on flat shapes snap_box is the right tool."""
    from layoutval.simulator import ClusterDisplay

    display = ClusterDisplay()
    display.state.update(
        {"TELLTALE_BATTERY_LOW": True, "TELLTALE_ABS": True, "FUEL_LEVEL": 0.6, "SPEED": 120}
    )
    cands = mser_candidates(display.render(), min_area=30)
    assert cands
    assert all(c.origin == "mser" for c in cands)
    assert all(c.bbox[2] > 0 and c.bbox[3] > 0 for c in cands)


def test_deduplicate_keeps_the_best_of_overlapping_candidates():
    kept = deduplicate([
        Proposal(bbox=(10, 10, 20, 20), score=5.0),
        Proposal(bbox=(11, 11, 20, 20), score=9.0),
        Proposal(bbox=(80, 80, 20, 20), score=1.0),
    ])
    assert len(kept) == 2
    assert kept[0].score == 9.0


def test_a_model_proposal_needs_a_human_before_it_becomes_an_element():
    class FakeSam:
        def propose(self, image, point):
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[30:60, 40:75] = 1
            return mask

    proposal = propose_with_model(FakeSam(), _scene(), (55, 45), name="sam2")
    assert proposal is not None
    assert proposal.origin == "sam2"

    with pytest.raises(UnconfirmedProposal):
        accept_proposal(proposal, "TELLTALE_X", confirmed_by="")
    with pytest.raises(UnconfirmedProposal):
        accept_proposal(proposal, "TELLTALE_X", confirmed_by="   ")

    spec = accept_proposal(proposal, "TELLTALE_X", confirmed_by="a.engineer",
                           kind=ElementKind.TELLTALE)
    assert spec.source == "manual"
    assert spec.kind is ElementKind.TELLTALE
    # The provenance has to survive into the report.
    assert "sam2" in spec.notes and "a.engineer" in spec.notes


def test_a_model_that_returns_nothing_yields_no_proposal():
    class EmptySam:
        def propose(self, image, point):
            return np.zeros(image.shape[:2], np.uint8)

    assert propose_with_model(EmptySam(), _scene(), (10, 10)) is None

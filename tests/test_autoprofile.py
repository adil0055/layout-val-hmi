"""What the inventory taken from a reference frame leaves out, and why."""

from __future__ import annotations

import cv2
import numpy as np

from layoutval.autoprofile import profile_from_reference

def test_a_straight_line_is_not_an_element():
    """Compared with itself, a 370 px edge matched 8 px down its own length."""
    from layoutval.autoprofile import structure

    img = np.zeros((200, 400), np.uint8)
    cv2.line(img, (20, 100), (380, 100), 255, 3)
    cv2.putText(img, "88", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    assert structure(img, (18, 97, 364, 7)) < 0.05
    assert structure(img, (150, 30, 60, 35)) > 0.12


def test_nothing_touching_the_edge_of_the_photograph_is_an_element():
    frame = np.full((300, 600, 3), 15, np.uint8)
    cv2.putText(frame, "km/h", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    cv2.putText(frame, "88", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    valid = np.ones((300, 600), bool)
    valid[:, :40] = False               # the corners were set past the photo's edge here
    ids = [e.id for e in profile_from_reference(frame, valid=valid)]
    assert len(ids) == 1 and ids[0].startswith("auto@25")

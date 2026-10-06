"""A camera held still: the same faulted screen captured again and again gives one answer.

What the bench showed: a webcam on a stand, nothing changed, and the answer
changed between captures -- the pose re-solved from what was drawn, tipped by
a capture's own noise towards a group that moved, and a smear estimated with a
moved group's ghost in it shifting the whole reference.
"""

from __future__ import annotations

import cv2
import numpy as np

from layoutval.blur import match_shake

from test_moving_phone import Bench, jpeg, start


def test_a_still_camera_takes_its_pose_from_the_screen_outline():
    bench = Bench()
    session = start(bench)
    report = session.handle("validate", jpeg(bench.shoot())).report
    assert any(f["flag"] == "pose_held" for f in report["flags"])
    assert report["verdict"] == "PASS"

    bench.drawing = dict(moved_block=(3.0, 0.0))
    report = session.handle("validate", jpeg(bench.shoot())).report
    assert any(f["flag"] == "pose_held" for f in report["flags"])
    moved = [e for e in report["elements"] if e["measurement"]["abs_delta"] > 1.0]
    assert moved and all(e["verdict"] == "FAIL" for e in moved)
    assert all(2.5 < e["measurement"]["abs_delta"] < 3.5 for e in moved)


def test_a_camera_moved_between_shots_is_not_held():
    bench = Bench()
    session = start(bench)
    bench.move(t=(0.08, 0, 0))
    report = session.handle("validate", jpeg(bench.shoot())).report
    assert not any(f["flag"] == "pose_held" for f in report["flags"])
    assert report["verdict"] == "PASS"


def _dots(shift=(0.0, 0.0), moved=()):
    """A grid of marks; those in ``moved`` drawn ``shift`` px off."""
    img = np.full((600, 900), 20, np.uint8)
    k = 0
    for y in range(60, 560, 70):
        for x in range(60, 860, 90):
            dx, dy = shift if k in moved else (0.0, 0.0)
            cv2.putText(img, "88", (int(round(x + dx)), int(round(y + dy))),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 230, 2, cv2.LINE_AA)
            k += 1
    return img


def test_a_group_that_moved_is_not_taken_for_smear():
    """Without the moved group named, the smear picks up its displacement as a
    ghost and its centring shifts every mark that did not move."""
    moved = set(range(0, 72, 2))                     # half the marks, 6 px right
    ref = _dots()
    live = cv2.GaussianBlur(_dots((6.0, 0.0), moved), (0, 0), 1.2)

    def shift_of_unmoved(m):
        # Where an unmoved mark sits in the matched reference, against the live.
        a, b = m.reference.astype(np.float32), m.live.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(a[40:80, 150:240], b[40:80, 150:240])
        return float(np.hypot(dx, dy))

    assert shift_of_unmoved(match_shake(ref, live)) > 1.0      # as it was: 3.2 px
    named = match_shake(ref, live, moved=[(6.0, 0.0)])
    assert named.kernel is not None
    assert shift_of_unmoved(named) < 0.3

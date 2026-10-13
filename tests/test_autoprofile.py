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


def test_a_soft_lit_patch_of_a_bar_is_not_an_element():
    """Glare changed the look of such patches and the matcher slid along the bar:
    5-8 px "wrong content" on a bench photograph of a screen that had not changed."""
    frame = np.full((300, 700, 3), 20, np.uint8)
    for label, x in (("88", 60), ("km", 200), ("PRND", 330)):
        cv2.putText(frame, label, (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 3)
    yy, xx = np.mgrid[0:300, 0:700].astype(np.float32)
    patch = 200 * np.exp(-(((xx - 520) / 22) ** 2 + ((yy - 207) / 7) ** 2))   # bright, but soft
    frame = np.clip(frame + patch[..., None], 0, 255).astype(np.uint8)
    # Bright enough to be segmented -- the old rule kept it...
    assert len(profile_from_reference(frame, min_structure=0.05, min_crispness=0.0)) == 4
    # ...and it carries no fine detail to measure, so it is left out.
    ids = [e.id for e in profile_from_reference(frame)]
    assert ids == ["auto@61,76", "auto@202,76", "auto@332,76"]


def test_tick_marks_are_elements_and_hairlines_are_not():
    """A speedometer's ticks (48-66 px) fell under the old 80 px floor; ticks a
    few pixels thick are narrower than a hand-held photo's smear, and stay out."""
    frame = np.full((400, 600, 3), 20, np.uint8)
    cv2.putText(frame, "88", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (230, 230, 230), 3)
    for x in (100, 180, 420):
        cv2.rectangle(frame, (x, 100), (x + 11, 115), (220, 220, 220), -1)    # 12x16: a chunky tick
    for x in (300, 500):
        cv2.rectangle(frame, (x, 300), (x + 3, 322), (220, 220, 220), -1)     # 4x23: a hairline tick
    boxes = [tuple(map(int, e.bbox)) for e in profile_from_reference(frame)]
    assert sum(1 for b in boxes if b[1] == 100) == 3
    assert not any(b[1] == 300 for b in boxes)


def test_what_hugs_the_frame_edge_is_not_an_element():
    """The display's rim, or a laptop's status icons: 1.6-2.7 px off under shake."""
    frame = np.full((400, 600, 3), 20, np.uint8)
    cv2.putText(frame, "88", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (230, 230, 230), 3)
    cv2.rectangle(frame, (570, 3), (590, 20), (230, 230, 230), -1)             # a corner icon
    boxes = [tuple(map(int, e.bbox)) for e in profile_from_reference(frame)]
    assert len(boxes) == 1 and boxes[0][0] > 200


def test_a_dials_ticks_are_kept_on_a_steady_camera_and_a_long_line_is_not():
    """A real webcam's ticks scored 0.035-0.08 against the straight-line cut of
    0.10: their ends fix them, but against their long sides they score low."""
    from layoutval.autoprofile import segment_reference, structure

    frame = np.full((400, 700, 3), 12, np.uint8)
    for label, x in (("88", 60), ("120", 420), ("km/h", 250)):
        cv2.putText(frame, label, (x, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (225, 225, 225), 3,
                    cv2.LINE_AA)
    for k, a in enumerate(range(60, 166, 15)):
        t = np.deg2rad(a)
        cx, cy, d = 90 + 70 * k, 140, np.array([np.cos(t), np.sin(t)]) * 26
        cv2.line(frame, tuple((np.array([cx, cy]) - d).astype(int)),
                 tuple((np.array([cx, cy]) + d).astype(int)), (215, 215, 215), 4, cv2.LINE_AA)
    cv2.line(frame, (60, 230), (640, 230), (215, 215, 215), 5, cv2.LINE_AA)
    frame = cv2.GaussianBlur(frame, (0, 0), 1.6)
    steady = dict(min_thickness_px=4, min_area_px=20)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ticks = [r for r in segment_reference(frame, **steady) if r[1] < 200]
    assert len(ticks) == 8 and all(0.03 < structure(gray, r[:4]) < 0.10 for r in ticks)

    def rows(p):
        return sorted(int(e.bbox[1]) // 100 for e in p.elements)

    # Hand-held rules: neither the ticks nor the line.
    assert rows(profile_from_reference(frame, **steady)) == [3, 3, 3]
    # A camera on a stand: the ticks, and still not the line.
    assert rows(profile_from_reference(frame, short_strokes=True, **steady)) == [1] * 8 + [3, 3, 3]

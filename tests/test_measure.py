"""Measurement primitives, pinned against synthetic ground truth."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from layoutval.measure import (
    angle_difference_deg,
    centroid_from_mask,
    needle_pose,
    phase_shift,
    subpixel_crop,
    subpixel_peak,
    zncc_match,
)


def _texture(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 1.6)


def test_phase_sign_convention():
    """Pins the sign: content at u in ref appears at u + (dx, dy) in live.

    The design notes say to check this against a known synthetic shift before
    trusting the direction, because getting it backwards produces numbers that
    look entirely reasonable and point the wrong way.
    """
    base = _texture()
    for sx, sy in ((3.0, 2.0), (-2.0, 1.0), (0.0, -4.0)):
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(base, M, (256, 256), flags=cv2.INTER_CUBIC)
        (dx, dy), _ = phase_shift(base[64:192, 64:192], shifted[64:192, 64:192])
        assert math.copysign(1, dx) == math.copysign(1, sx) or sx == 0
        assert abs(dx - sx) < 0.5, (sx, dx)
        assert abs(dy - sy) < 0.5, (sy, dy)


def test_phase_shift_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        phase_shift(np.zeros((8, 8), np.float32), np.zeros((8, 9), np.float32))


def test_subpixel_peak_returns_none_on_border():
    """A peak on the search edge is a failure of unknown magnitude, not a pass."""
    res = np.zeros((5, 5), np.float32)
    res[0, 2] = 1.0
    assert subpixel_peak(res, 2, 0) is None
    res = np.zeros((5, 5), np.float32)
    res[2, 4] = 1.0
    assert subpixel_peak(res, 4, 2) is None


def test_subpixel_peak_interpolates():
    res = np.zeros((5, 5), np.float32)
    res[2, 1], res[2, 2], res[2, 3] = 0.6, 1.0, 0.8
    res[1, 2], res[3, 2] = 0.9, 0.9
    out = subpixel_peak(res, 2, 2)
    assert out is not None
    x, y = out
    assert 2.0 < x < 2.5  # pulled towards the higher right-hand neighbour
    assert y == pytest.approx(2.0, abs=1e-6)


def test_subpixel_peak_does_not_extrapolate_outside_its_samples():
    """A flat neighbourhood must not produce a confident fractional answer."""
    res = np.full((5, 5), 0.5, np.float32)
    res[2, 2] = 0.50001
    res[2, 3] = 0.5
    x, y = subpixel_peak(res, 2, 2)
    assert abs(x - 2) <= 1.0 and abs(y - 2) <= 1.0


def test_subpixel_crop_anchors_at_fractional_coordinate():
    img = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    assert subpixel_crop(img, (10.0, 20.0, 5, 5))[0, 0] == pytest.approx(img[20, 10])
    assert subpixel_crop(img, (10.5, 20.0, 5, 5))[0, 0] == pytest.approx(
        (img[20, 10] + img[20, 11]) / 2
    )


def test_zncc_match_finds_a_known_offset():
    area = np.zeros((80, 80), np.float32)
    cv2.circle(area, (40, 30), 9, 1.0, -1)
    template = np.zeros((30, 30), np.float32)
    cv2.circle(template, (15, 15), 9, 1.0, -1)
    (x, y), score, border = zncc_match(area, template)
    assert not border
    assert score > 0.9
    assert x == pytest.approx(25, abs=0.6)
    assert y == pytest.approx(15, abs=0.6)


def test_zncc_match_rejects_template_larger_than_search_area():
    with pytest.raises(ValueError):
        zncc_match(np.zeros((10, 10), np.float32), np.zeros((20, 20), np.float32))


def test_centroid_from_mask():
    mask = np.zeros((40, 40), np.uint8)
    cv2.rectangle(mask, (10, 10), (19, 19), 255, -1)
    (cx, cy), area = centroid_from_mask(mask)
    assert cx == pytest.approx(14.5, abs=0.01)
    assert cy == pytest.approx(14.5, abs=0.01)
    assert area / 255.0 == pytest.approx(100.0)
    assert centroid_from_mask(np.zeros((10, 10), np.uint8)) is None


def test_angle_difference_wraps():
    assert angle_difference_deg(179, -179) == pytest.approx(-2)
    assert angle_difference_deg(-179, 179) == pytest.approx(2)
    assert angle_difference_deg(10, 350) == pytest.approx(20)


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, -45, -120])
def test_needle_pose_recovers_pivot_and_angle(angle):
    """The hub must not bias the pivot.

    A needle's tail extreme sits a hub radius past the true pivot -- several
    pixels of systematic error that averaging never removes.
    """
    mask = np.zeros((240, 240), np.uint8)
    pivot = (120.0, 120.0)
    a = math.radians(angle)
    tip = (pivot[0] + 80 * math.cos(a), pivot[1] - 80 * math.sin(a))
    cv2.line(mask, (120, 120), (int(round(tip[0])), int(round(tip[1]))), 255, 5, cv2.LINE_AA)
    cv2.circle(mask, (120, 120), 9, 255, -1, cv2.LINE_AA)
    mask = (mask > 127).astype(np.uint8) * 255

    pose = needle_pose(mask, expected_pivot=pivot)
    assert pose is not None
    assert pose.hub_found
    assert math.hypot(pose.pivot[0] - pivot[0], pose.pivot[1] - pivot[1]) < 1.0
    assert abs(angle_difference_deg(pose.angle_deg, angle)) < 1.5


def test_needle_pose_reports_when_there_is_no_hub():
    mask = np.zeros((200, 200), np.uint8)
    cv2.line(mask, (100, 100), (170, 100), 255, 4)
    pose = needle_pose(mask, expected_pivot=(100.0, 100.0))
    assert pose is not None
    assert not pose.hub_found  # so the consumer knows the pivot carries a bias


def test_needle_pose_returns_none_on_empty_mask():
    assert needle_pose(np.zeros((50, 50), np.uint8), (25.0, 25.0)) is None


def test_an_element_touching_the_frame_edge_is_measured_like_any_other():
    """Compared with itself, an element in the frame's corner came back
    "missing or displaced" at 0.00 px: its search window, clipped by the edge,
    began exactly where it sat."""
    import cv2

    from layoutval.measure import measure_translation
    from layoutval.types import ElementSpec, PositionModel
    from layoutval.verdict import verdict_for

    img = np.full((400, 900, 3), 12, np.uint8)
    cv2.ellipse(img, (0, 0), (60, 60), 0, 0, 90, (200, 200, 200), 3)
    cv2.ellipse(img, (899, 399), (60, 60), 0, 180, 270, (200, 200, 200), 3)
    for bbox in [(0.0, 0.0, 62.0, 62.0), (837.0, 337.0, 63.0, 63.0)]:
        spec = ElementSpec(id="edge", bbox=bbox, position=PositionModel(origin=bbox[:2]))
        same = measure_translation(img, img.copy(), spec)
        assert verdict_for(same, spec.tolerance)[0].value == "PASS"
        assert abs(same.dx) < 0.05 and abs(same.dy) < 0.05
        shifted = cv2.warpAffine(img, np.float32([[1, 0, 2], [0, 1, 1]]), (900, 400),
                                 borderMode=cv2.BORDER_REPLICATE)
        moved = measure_translation(img, shifted, spec)
        assert abs(moved.dx - 2.0) < 0.1 and abs(moved.dy - 1.0) < 0.1

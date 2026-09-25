"""The residual check and the z-order assertion: the two catches for what
per-element position checks cannot see."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval.residual import check_occlusions, dissimilarity_heatmap, residual_check
from layoutval.types import ElementSpec, PositionModel, Tolerance, Verdict


def _canvas() -> np.ndarray:
    img = np.full((200, 300, 3), 12, np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), (200, 200, 200), -1)
    cv2.circle(img, (200, 100), 25, (60, 160, 220), -1)
    return img


def test_identical_frames_produce_no_findings():
    img = _canvas()
    score, findings = residual_check(img, img.copy())
    assert score > 0.999
    assert findings == []


def test_an_unmodelled_artefact_is_found_and_marked_as_overlapping_nothing():
    ref = _canvas()
    live = ref.copy()
    cv2.rectangle(live, (140, 20), (180, 45), (40, 150, 225), -1)
    specs = [ElementSpec(id="known", bbox=(30.0, 30.0, 60.0, 60.0))]
    score, findings = residual_check(ref, live, specs)
    assert score < 1.0
    orphans = [f for f in findings if not f.overlaps]
    assert orphans
    hit = max(orphans, key=lambda f: f.area_px)
    assert abs(hit.bbox[0] - 140) < 8 and abs(hit.bbox[1] - 20) < 8


def test_findings_are_attributed_to_the_element_they_overlap():
    ref = _canvas()
    live = ref.copy()
    cv2.rectangle(live, (30, 30), (90, 90), (12, 12, 12), -1)  # the square vanishes
    specs = [ElementSpec(id="square", bbox=(30.0, 30.0, 60.0, 60.0))]
    _, findings = residual_check(ref, live, specs)
    assert any("square" in f.overlaps for f in findings)


def test_findings_use_the_moving_element_s_expected_box_for_the_state():
    ref = _canvas()
    live = ref.copy()
    cv2.rectangle(live, (150, 150), (190, 180), (90, 200, 90), -1)
    spec = ElementSpec(
        id="bar",
        signal="LEVEL",
        bbox=(10.0, 150.0, 40.0, 30.0),
        position=PositionModel(kind="linear", origin=(10.0, 150.0),
                               direction=(140.0, 0.0), value_min=0.0, value_max=1.0),
    )
    _, findings = residual_check(ref, live, [spec], values={"LEVEL": 1.0})
    assert any("bar" in f.overlaps for f in findings)
    _, findings_wrong_state = residual_check(ref, live, [spec], values={"LEVEL": 0.0})
    assert all("bar" not in f.overlaps for f in findings_wrong_state)


def test_residual_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        residual_check(np.zeros((10, 10, 3), np.uint8), np.zeros((12, 10, 3), np.uint8))


def test_heatmap_is_produced_for_review():
    ref = _canvas()
    live = ref.copy()
    cv2.rectangle(live, (140, 20), (180, 45), (40, 150, 225), -1)
    heat = dissimilarity_heatmap(ref, live)
    assert heat.shape == ref.shape
    assert heat[:, :, 0].std() > 0


def _draw_plate(img: np.ndarray) -> None:
    cv2.rectangle(img, (40, 40), (110, 110), (80, 80, 80), -1)
    for i in range(45, 110, 12):
        cv2.line(img, (42, i), (108, i), (150, 150, 150), 2)


def _draw_badge(img: np.ndarray) -> None:
    cv2.rectangle(img, (70, 70), (130, 130), (30, 180, 240), -1)
    cv2.circle(img, (100, 100), 18, (10, 40, 90), 3)
    cv2.line(img, (78, 78), (122, 122), (10, 40, 90), 2)


def _overlapping_pair() -> tuple[np.ndarray, list[ElementSpec]]:
    ref = np.full((160, 160, 3), 10, np.uint8)
    _draw_plate(ref)
    _draw_badge(ref)  # badge on top
    specs = [
        ElementSpec(id="badge", bbox=(70.0, 70.0, 60.0, 60.0), occludes=["plate"],
                    tolerance=Tolerance(identity_min=0.8)),
        ElementSpec(id="plate", bbox=(40.0, 40.0, 70.0, 70.0)),
    ]
    return ref, specs


def test_z_order_assertion_passes_when_the_order_is_right():
    ref, specs = _overlapping_pair()
    results = check_occlusions(ref, ref.copy(), specs)
    assert results and all(r.verdict is Verdict.PASS for r in results)
    assert results[0].element_id == "badge>plate"


def test_z_order_assertion_catches_a_swapped_order():
    """Both elements in the right place with the wrong one on top passes every
    per-element position check that can be written."""
    ref, specs = _overlapping_pair()
    live = np.full((160, 160, 3), 10, np.uint8)
    _draw_badge(live)
    _draw_plate(live)  # plate now on top
    results = check_occlusions(ref, live, specs)
    assert results[0].verdict is Verdict.FAIL
    assert results[0].reason == "z_order"


def test_z_order_assertion_refuses_to_judge_a_flat_overlap():
    """Two flat patches correlate perfectly whatever their colours."""
    ref = np.full((160, 160, 3), 10, np.uint8)
    cv2.rectangle(ref, (40, 40), (110, 110), (80, 80, 80), -1)
    cv2.rectangle(ref, (70, 70), (130, 130), (30, 180, 240), -1)
    specs = [
        ElementSpec(id="badge", bbox=(70.0, 70.0, 60.0, 60.0), occludes=["plate"]),
        ElementSpec(id="plate", bbox=(40.0, 40.0, 70.0, 70.0)),
    ]
    results = check_occlusions(ref, ref.copy(), specs)
    assert results[0].verdict is Verdict.REVIEW
    assert results[0].reason == "occlusion_indeterminate"


def test_z_order_assertion_says_so_when_the_boxes_do_not_overlap():
    ref = np.full((160, 160, 3), 10, np.uint8)
    specs = [
        ElementSpec(id="a", bbox=(0.0, 0.0, 20.0, 20.0), occludes=["b"]),
        ElementSpec(id="b", bbox=(100.0, 100.0, 20.0, 20.0)),
    ]
    results = check_occlusions(ref, ref.copy(), specs)
    assert results[0].verdict is Verdict.REVIEW
    assert results[0].reason == "occlusion_not_applicable"


def test_a_faint_texture_difference_is_not_a_finding():
    """On a dark flat ground SSIM calls any texture change total; it is not a change."""
    from layoutval.residual import residual_check

    rng = np.random.default_rng(0)
    ref = np.full((300, 400, 3), 16, np.uint8)
    live = ref.copy()
    # Moire-like ripple, a dozen levels deep: what re-photographing a panel does.
    yy, xx = np.mgrid[0:300, 0:400]
    ripple = (6 * np.sin(xx / 3.1) * np.sin(yy / 2.7)).astype(np.float32)
    live = np.clip(live + ripple[..., None] + rng.normal(0, 2, live.shape), 0, 255).astype(np.uint8)
    _, faint = residual_check(ref, live)
    assert faint == []
    cv2.rectangle(live, (200, 120), (246, 142), (230, 230, 230), -1)   # something new drawn
    _, found = residual_check(ref, live)
    assert found and any(abs(f.bbox[0] - 200) < 8 for f in found)
    assert all(f.difference >= 40 for f in found)

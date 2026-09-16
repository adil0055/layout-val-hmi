"""Capture: stacking, settling, and refusing to measure a frame in motion."""

from __future__ import annotations

import numpy as np
import pytest

from layoutval.capture import (
    CallableSource,
    DirectorySource,
    SettlingTimeout,
    estimate_noise_floor,
    median_stack,
    settled_median,
    to_gray,
    wait_for_settle,
)


def test_median_stack_rejects_outlier_frames():
    """Median, not mean: one bad frame must not be averaged in."""
    frames = [np.full((4, 4), 10, np.uint8) for _ in range(5)]
    frames[2] = np.full((4, 4), 200, np.uint8)
    assert np.all(median_stack(frames) == 10)


def test_median_stack_needs_a_frame():
    with pytest.raises(ValueError):
        median_stack([])


def test_estimate_noise_floor_measures_temporal_spread():
    rng = np.random.default_rng(0)
    frames = [
        np.clip(100 + rng.normal(0, 3.0, (64, 64)), 0, 255).astype(np.uint8)
        for _ in range(30)
    ]
    assert estimate_noise_floor(frames) == pytest.approx(3.0, abs=0.6)


def test_wait_for_settle_returns_once_the_screen_is_still():
    frame = np.full((32, 32, 3), 40, np.uint8)
    result = wait_for_settle(CallableSource(lambda: frame), stable_frames=3, timeout_s=2.0)
    assert result.settled
    assert result.last_diff < 1.0


def test_settling_timeout_fails_the_case_rather_than_measuring_mid_transition():
    """Measuring whatever was on screen when time ran out is the failure mode
    this exists to prevent."""
    state = {"i": 0}

    def moving() -> np.ndarray:
        state["i"] += 1
        img = np.zeros((32, 32, 3), np.uint8)
        img[:, state["i"] % 32] = 255
        return img

    with pytest.raises(SettlingTimeout):
        wait_for_settle(CallableSource(moving), stable_frames=3, timeout_s=0.4)

    result = wait_for_settle(
        CallableSource(moving), stable_frames=3, timeout_s=0.4, raise_on_timeout=False
    )
    assert not result.settled


def test_settled_median_combines_both():
    frame = np.full((16, 16, 3), 77, np.uint8)
    out = settled_median(CallableSource(lambda: frame), n=5, stable_frames=2, timeout_s=1.0)
    assert np.all(out == 77)


def test_directory_source_round_trips(tmp_path):
    import cv2

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.full((8, 8, 3), i * 10, np.uint8))
    source = DirectorySource.from_glob(tmp_path)
    assert [int(source.read()[0, 0, 0]) for _ in range(3)] == [0, 10, 20]
    assert int(source.read()[0, 0, 0]) == 0  # loops


def test_to_gray_is_luma_not_a_single_channel():
    """Text drawn with sub-pixel anti-aliasing has per-channel centroid shifts;
    measuring on luma is what cancels them."""
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 1] = 200  # green only
    grey = to_gray(img)
    assert 100 < int(grey[0, 0]) < 130  # ~0.587 * 200
    assert to_gray(grey) is grey

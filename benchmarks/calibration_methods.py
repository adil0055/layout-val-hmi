#!/usr/bin/env python3
"""Can the display-to-camera mapping be solved without drawing a pattern?

Four ways to establish where the display is in the camera frame, scored against
each other the way that actually matters: rectify with each mapping, then
measure where the elements land compared with the true framebuffer. A tidy
homography residual says nothing about that; a quarter-pixel error in the
mapping is a quarter-pixel error in every measurement taken through it.

    python benchmarks/calibration_methods.py

The short version of what this measures: **matching the screen's own content
against its framebuffer is as accurate as a chessboard**, provided the feature
fit is refined. The feature fit alone ties the chessboard at matched scale and
falls apart at 0.7 px when the render and the photograph differ in scale, which
is exactly the case an under-sampled rig is in.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import (  # noqa: E402
    PATTERN,
    PATTERN_ORIGIN,
    SQUARE_PX,
    build_bench,
)

from layoutval.autoprofile import profile_from_reference  # noqa: E402
from layoutval.calibration import (  # noqa: E402
    DisplayGeometry,
    chessboard_display_points,
    homography_from_display_pattern,
    homography_from_screen_content,
)
from layoutval.measure import measure_translation  # noqa: E402


def element_rms(truth, frame, H, probe, size):
    """Where the elements land after rectifying, in display px."""
    if H is None:
        return None
    rectified = DisplayGeometry(H=H, display_size=size).rectify(frame)
    errors = [
        float(np.hypot(m.dx, m.dy))
        for spec in probe
        for m in [measure_translation(truth, rectified, spec)]
        if m.dx is not None and m.zncc and m.zncc > 0.5
    ]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def chessboard(bench, size):
    bench.rig.show("checkerboard")
    board = bench.undistort(bench.rig.read())
    bench.rig.show("main")
    try:
        return homography_from_display_pattern(
            board, PATTERN,
            chessboard_display_points(PATTERN, SQUARE_PX, PATTERN_ORIGIN),
            display_size=size,
        ).H
    except RuntimeError:
        return None


def content(truth, frame, size, refine):
    try:
        return homography_from_screen_content(
            truth, frame, display_size=size, refine=refine
        ).H
    except RuntimeError:
        return None


def run(label, **camera):
    bench = build_bench(**camera)
    bench.reset()
    size = bench.display.size
    truth = bench.display.render()
    frame = bench.undistort(bench.rig.read())
    probe = profile_from_reference(truth, screen="probe", display_size=size)

    columns = [
        chessboard(bench, size),
        content(truth, frame, size, refine=False),
        content(truth, frame, size, refine=True),
    ]
    cells = "".join(
        "  failed" if (v := element_rms(truth, frame, H, probe, size)) is None
        else f"{v:9.3f}"
        for H in columns
    )
    print(f"{label:<32}{cells}")


def main() -> int:
    print("Where elements land after rectifying, in display px (lower is better)\n")
    print(f"{'condition':<32}{'chessboard':>9}{'content':>9}{'+refined':>9}")
    print("-" * 59)
    run("baseline")
    run("steep angle", tilt_deg=12.0, roll_deg=5.0)
    run("noisy (sigma 8)", noise_sigma=8.0)
    run("soft focus (defocus 2.5)", defocus_sigma=2.5)
    run("under-sampled (ratio 1.0)", sampling_ratio=1.0)
    run("well-sampled (ratio 3.0)", sampling_ratio=3.0, sensor_size=(3000, 1400))
    run("heavy distortion", k1=-0.25, k2=0.08)
    print()
    print("'content' is SIFT and RANSAC against the framebuffer; '+refined' adds")
    print("ECC in homography mode, seeded from it. The seed alone is enough only")
    print("when the render and the photograph are at a similar scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

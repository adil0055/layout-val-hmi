#!/usr/bin/env python3
"""Which sub-pixel shift estimator is actually more accurate, measured.

The received advice is that phase correlation is the primary shift estimate and
normalised cross-correlation is for identity confirmation.  On instrument-cluster
content this implementation measures the opposite, consistently, across imaging
regimes.  This script is that measurement, so the claim can be re-run rather than
taken on trust -- including on your own content, by pointing it at a different
display renderer.

Two things are measured, and they are not the same thing:

* **accuracy** -- error against a known injected displacement, which is what a
  tolerance is actually about;
* **peak locking** -- how that error varies with the *fractional* part of the
  displacement.  It is invisible to a static-screen repeatability study, and on
  an under-sampled rig it is usually the largest error term.

Run:  python benchmarks/estimator_comparison.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from layoutval.calibration import (  # noqa: E402
    Intrinsics,
    Undistorter,
    chessboard_display_points,
    homography_from_display_pattern,
)
from layoutval.capture import capture, median_stack  # noqa: E402
from layoutval.measure import measure_translation  # noqa: E402
from layoutval.profile import import_design_tree  # noqa: E402
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera  # noqa: E402
from layoutval.types import PositionModel, Tolerance  # noqa: E402

TARGETS = ["TELLTALE_ABS", "LABEL_SPEED_UNITS", "GAUGE_PLATE", "FUEL_BAR"]
NOMINAL = {
    "TELLTALE_BATTERY_LOW": True,
    "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True,
    "FUEL_LEVEL": 0.6,
    "SPEED": 120.0,
}
SHIFTS = [round(i * 0.25, 2) for i in range(13)]
EXPORT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "design_export.json"


def build(**camera_kwargs):
    display = ClusterDisplay()
    camera = VirtualCamera(display_size=display.size, **camera_kwargs)
    rig = SimulatedRig(display, camera)
    undistort = Undistorter(
        Intrinsics(K=camera.K, dist=camera.dist, image_size=camera.sensor_size)
    )
    rig.show("checkerboard")
    geometry = homography_from_display_pattern(
        undistort(median_stack(capture(rig, n=7))),
        (9, 6),
        chessboard_display_points((9, 6), 40, (60, 20)),
        display_size=display.size,
    )
    rig.show("main")
    display.state.update(NOMINAL)

    profile = import_design_tree(
        json.loads(EXPORT.read_text()),
        screen="main",
        display_size=display.size,
        defaults=Tolerance(),
    )
    profile["FUEL_BAR"].position = PositionModel(
        kind="linear", origin=(120.0, 300.0), direction=(300.0, 0.0),
        value_min=0.0, value_max=1.0,
    )

    def grab():
        return geometry.rectify(undistort(median_stack(capture(rig, n=9))))

    return display, profile, grab


def sweep(label: str, **camera_kwargs) -> dict[str, list[float]]:
    display, profile, grab = build(**camera_kwargs)
    reference = grab()
    errors: dict[str, list[float]] = {"zncc": [], "phase": []}
    for ox in SHIFTS:
        for t in TARGETS:
            display.offsets[t] = (ox, 0.0)
        live = grab()
        for t in TARGETS:
            m = measure_translation(
                reference, live, profile[t], value=0.6 if t == "FUEL_BAR" else None
            )
            if m.zncc_dx is not None:
                errors["zncc"].append(np.hypot(m.zncc_dx - ox, m.zncc_dy))
            if m.phase_dx is not None:
                errors["phase"].append(np.hypot(m.phase_dx - ox, m.phase_dy))
    z, p = np.array(errors["zncc"]), np.array(errors["phase"])
    print(
        f"{label:<32} zncc rms={np.sqrt((z ** 2).mean()):.3f} max={z.max():.3f}   "
        f"phase rms={np.sqrt((p ** 2).mean()):.3f} max={p.max():.3f}"
    )
    return errors


def main() -> int:
    print("Accuracy against a known injected shift (display px, lower is better)\n")
    sweep("baseline (ratio 1.6)")
    sweep("noisy (sigma 6.0)", noise_sigma=6.0)
    sweep("under-sampled (ratio 1.0)", sampling_ratio=1.0)
    sweep("well-sampled (ratio 3.0)", sampling_ratio=3.0, sensor_size=(3000, 1400))
    sweep("soft focus (defocus 2.0)", defocus_sigma=2.0)

    print("\nPeak locking: error against the fractional part of the shift.")
    print("A static-screen repeatability study reports none of this.\n")
    display, profile, grab = build()
    reference = grab()
    spec = profile["FUEL_BAR"]
    print(f"  {'shift':>6}{'zncc err':>11}{'phase err':>11}")
    for ox in SHIFTS:
        display.offsets["FUEL_BAR"] = (ox, 0.0)
        m = measure_translation(reference, grab(), spec, value=0.6)
        print(f"  {ox:>6.2f}{m.zncc_dx - ox:>+11.3f}{m.phase_dx - ox:>+11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared fixtures: a calibrated simulated rig, built once per session."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layoutval.calibration import (  # noqa: E402
    Calibration,
    DisplayGeometry,
    Intrinsics,
    Undistorter,
    chessboard_display_points,
    homography_from_display_pattern,
)
from layoutval.capture import capture, median_stack  # noqa: E402
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera  # noqa: E402

PATTERN = (9, 6)
SQUARE_PX = 40
PATTERN_ORIGIN = (60, 20)

NOMINAL = {
    "TELLTALE_BATTERY_LOW": True,
    "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True,
    "FUEL_LEVEL": 0.6,
    "SPEED": 120.0,
}
VALUES = {"FUEL_LEVEL": 0.6, "SPEED": 120.0}


@dataclass
class Bench:
    display: ClusterDisplay
    camera: VirtualCamera
    rig: SimulatedRig
    undistort: Undistorter
    geometry: DisplayGeometry
    grab: Callable[[], np.ndarray]

    @property
    def calibration(self) -> Calibration:
        return Calibration(
            intrinsics=self.undistort.intrinsics,
            geometry=self.geometry,
            rig={"source": "simulator"},
        )

    def reset(self) -> None:
        self.display.offsets.clear()
        self.display.swapped.clear()
        self.display.hidden.clear()
        self.display.artefacts.clear()
        self.display.needle_angle_offset_deg = 0.0
        self.display.needle_pivot_offset = (0.0, 0.0)
        self.display.state.update(NOMINAL)


def build_bench(**camera_kwargs) -> Bench:
    display = ClusterDisplay()
    camera = VirtualCamera(display_size=display.size, **camera_kwargs)
    rig = SimulatedRig(display, camera)
    undistort = Undistorter(
        Intrinsics(K=camera.K, dist=camera.dist, image_size=camera.sensor_size)
    )
    rig.show("checkerboard")
    geometry = homography_from_display_pattern(
        undistort(median_stack(capture(rig, n=7))),
        PATTERN,
        chessboard_display_points(PATTERN, SQUARE_PX, PATTERN_ORIGIN),
        display_size=display.size,
    )
    rig.show("main")
    display.state.update(NOMINAL)

    def grab() -> np.ndarray:
        return geometry.rectify(undistort(median_stack(capture(rig, n=9))))

    return Bench(display, camera, rig, undistort, geometry, grab)


@pytest.fixture(scope="session")
def _bench() -> Bench:
    return build_bench()


@pytest.fixture()
def bench(_bench: Bench) -> Bench:
    _bench.reset()
    return _bench


@pytest.fixture(scope="session")
def _reference(_bench: Bench) -> np.ndarray:
    _bench.reset()
    return _bench.grab()


@pytest.fixture()
def reference(_reference: np.ndarray) -> np.ndarray:
    return _reference

"""How good is a board on the *bezel* instead of on the screen?

A production cluster will not draw a chessboard because you asked it to.  A
sticker on its bezel needs nothing from the build at all -- so the question is
what that costs against the on-screen board, which is the method everything
else in this package is measured against.

Two things came out of running this, and only one of them is about markers:

* **Undistortion is not optional.**  A homography cannot represent lens
  distortion, and the markers are photographed in a different part of the frame
  from the active area, so the board fit and the display fit are each locally
  wrong in a different direction and the errors compound rather than cancel.
  Undistorted, this route holds about 0.10 px whatever the lens.  With the
  distortion left in it runs 0.6 px to 5.7 px depending on how wide the lens
  is, and elements start failing to match at all.  Because the size of the
  error is a property of the lens, a frame that happens to look fine on one
  camera says nothing about the next -- so the server refuses the route without
  intrinsics rather than warning about it.
* **The anchor is only as good as the view it was measured in.**  Re-bound in
  the pose it is used in, the bezel board matches an on-screen board.  Carried
  across a deliberate camera move it degrades by roughly 5x -- still usable,
  no longer sub-tenth-pixel.  It degrades quietly, which is why the server
  compares the board's position against the binding frame and says so.

Caveat this fixture cannot settle: here the bezel and the active area are
coplanar, because both are drawn on one flat panel.  A real display is recessed
behind glass, and a single homography anchor is exact only for a coplanar pair.
Expect the real figure to be worse and measure it on the bench.
"""

from __future__ import annotations

import numpy as np

from layoutval.autoprofile import profile_from_reference
from layoutval.calibration import (
    CharucoSpec,
    DisplayGeometry,
    Intrinsics,
    Undistorter,
    charuco_anchor,
    chessboard_display_points,
    homography_from_charuco,
    homography_from_display_pattern,
)
from layoutval.measure import measure_translation
from layoutval.simulator import BezelPanel, BezelRig, ClusterDisplay, VirtualCamera

NOMINAL = {
    "TELLTALE_BATTERY_LOW": True,
    "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True,
    "FUEL_LEVEL": 0.6,
    "SPEED": 120.0,
}

PATTERN, SQUARE_PX, ORIGIN = (9, 6), 40.0, (60.0, 20.0)

#: Poses to average over.  A single pose is not a measurement: the first run of
#: this swept sampling ratio at one pose and came out non-monotonic, which was
#: pose scatter being read as a trend.
POSES = [(2.0, 0.7), (6.0, -3.0), (11.0, 2.5), (-4.0, 5.0), (8.0, -8.0),
         (0.5, 0.2), (14.0, 1.0), (-9.0, -4.0), (3.0, 9.0), (5.5, -1.5)]


def _rig(panel, ratio, tilt, roll, seed, **lens):
    display = ClusterDisplay()
    display.state.update(NOMINAL)
    return BezelRig(display, panel, VirtualCamera(
        display_size=panel.panel_size, sensor_size=(2400, 1500),
        sampling_ratio=ratio, tilt_deg=tilt, roll_deg=roll, seed=seed, **lens))


def _rms_error(rig, geometry, truth, probe, live):
    rect = DisplayGeometry(H=geometry, display_size=rig.display.size).rectify(live)
    errors = [
        float(np.hypot(m.dx, m.dy))
        for spec in probe
        for m in [measure_translation(truth, rect, spec)]
        if m.dx is not None and m.zncc and m.zncc > 0.5
    ]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def trial(ratio, tilt, roll, seed, *, rebind: bool, undistort: bool = True,
          k1: float | None = None, k2: float | None = None):
    """One pose.  ``rebind`` binds the anchor in the pose it is then used in."""
    panel = BezelPanel()
    lens = {} if k1 is None else {"k1": k1, "k2": k2}
    rig = _rig(panel, ratio, tilt, roll, seed, **lens)
    undistorter = Undistorter(Intrinsics(
        K=rig.camera.K, dist=rig.camera.dist, image_size=rig.camera.sensor_size))
    grab = (lambda r: undistorter(r.read())) if undistort else (lambda r: r.read())

    truth = rig.display.render()
    probe = profile_from_reference(truth, screen="bench", display_size=rig.display.size)
    board = CharucoSpec(panel.squares[0], panel.squares[1], panel.square_length,
                        panel.square_length * panel.marker_ratio).board()
    points = chessboard_display_points(PATTERN, SQUARE_PX, ORIGIN)

    bind_rig = rig if rebind else _rig(panel, ratio, 2.0, 0.7, 1000 + seed, **lens)
    bind_rig.show("checkerboard")
    bind_frame = grab(bind_rig)
    bind_rig.show("main")
    bound = homography_from_display_pattern(
        bind_frame, PATTERN, points, display_size=rig.display.size)
    anchor = charuco_anchor(bind_frame, board, bound)

    live = grab(rig)
    markers = homography_from_charuco(
        live, board, anchor, display_size=rig.display.size)

    rig.show("checkerboard")
    on_screen = homography_from_display_pattern(
        grab(rig), PATTERN, points, display_size=rig.display.size)
    rig.show("main")

    return (_rms_error(rig, on_screen.H, truth, probe, live),
            _rms_error(rig, markers.H, truth, probe, live))


def main() -> None:
    print(__doc__.strip().split("\n\n")[0])
    print()
    for ratio in (0.78, 0.95):
        for rebind, label in ((True, "anchor re-bound in the pose it is used in"),
                              (False, "anchor bound once, camera then moved")):
            screen, markers = [], []
            for i, (tilt, roll) in enumerate(POSES):
                try:
                    a, b = trial(ratio, tilt, roll, i, rebind=rebind)
                except RuntimeError as exc:
                    print(f"    pose {tilt},{roll}: {exc}")
                    continue
                screen.append(a)
                markers.append(b)
            fmt = (lambda v: f"p50 {np.median(v):6.3f}  p95 "
                             f"{np.percentile(v, 95):6.3f}  max {max(v):6.3f}")
            print(f"sampling ratio {ratio}, {label}  (n={len(screen)})")
            print(f"   board on the screen   {fmt(screen)}")
            print(f"   board on the bezel    {fmt(markers)}")
            print()

    # Why undistortion is refused rather than warned about: the penalty is a
    # property of the lens, not of the shot.
    print("and what leaving the distortion in costs, by lens:")
    print(f"   {'lens':<24}{'undistorted':>12}{'distortion left in':>21}")
    for label, k1, k2 in (("mild (k1=-0.03)", -0.03, 0.005),
                          ("default (k1=-0.09)", -0.09, 0.02),
                          ("wide (k1=-0.25)", -0.25, 0.08),
                          ("very wide (k1=-0.45)", -0.45, 0.20)):
        cells = []
        for undistort in (True, False):
            values = []
            for i, (tilt, roll) in enumerate(POSES[:5]):
                try:
                    values.append(trial(0.95, tilt, roll, i, rebind=True,
                                        undistort=undistort, k1=k1, k2=k2)[1])
                except RuntimeError:
                    pass
            cells.append(f"{np.median(values):6.3f} px" if values else "   n/a  ")
        print(f"   {label:<24}{cells[0]:>12}{cells[1]:>21}")
    print()
    print("   Elements also drop out of the match entirely on the wider lenses,")
    print("   which is the part that does not show up as a bigger number.")


if __name__ == "__main__":
    main()

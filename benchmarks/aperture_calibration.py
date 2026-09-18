"""Calibrating from the display's own border, with nothing drawn for the camera.

A production cluster shows what it shows.  It will not draw a chessboard, it
will not hand over its framebuffer, and it will not hold still on a calibration
screen while you bind a bezel marker to it.  This measures the route that asks
it for none of those: find the display's physical opening in the trim, fit its
four edges, and take that as display space.

Four things came out of building it, and three were mistakes worth recording:

* **Otsu does not work.**  A photograph of a cluster has at least three
  populations -- dark screen, mid trim, something bright -- and a two-class
  split lands between the bright thing and everything else.  Measured, it put
  the threshold at 123 and merged screen and trim into one blob covering 99.9%
  of the frame.  The threshold is swept instead, and the aperture is the region
  that keeps the right shape across the widest range of levels.
* **The aperture is a hole.**  ``RETR_EXTERNAL`` discards it by definition, and
  taken the other way round the dark screen merges with the dark room behind
  the cluster.  That one flag was the difference between finding the aperture
  in every pose and in none.
* **Half a pixel is not nothing.**  The opening is the *outer* boundary of the
  edge pixels, so it spans -0.5 to w-0.5 and not 0 to w.  Getting it wrong
  shifts every pose identically, which is exactly the kind of error that looks
  like success: 0.70 px, which is 0.5 px on each axis, hiding as a constant.
* **A coarse answer is worse than none.**  Falling back to the threshold
  corners when sub-pixel refinement failed seemed obviously right -- they are
  worth about a pixel, and a usable answer beats a refusal.  Measured, every
  frame that fell back was 1.7 to 3.1 px out while every refined frame was
  inside 0.05 px, and nothing in the result distinguished them.  It refuses now.

What it will not tell you: the opening is the physical aperture, and the active
area sits behind it by a mask width that is invisible from outside.  So this
solves a frame carrying a constant offset against true display coordinates.
That offset cancels exactly between reference and validate -- both rectify
through the same homography -- so it does not touch a defect measurement.  It
matters only for comparing against a design in absolute coordinates.
"""

from __future__ import annotations

import numpy as np

from layoutval.autoprofile import profile_from_reference
from layoutval.calibration import (
    DisplayGeometry,
    Intrinsics,
    Undistorter,
    chessboard_display_points,
    homography_from_display_aperture,
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
POSES = [(2.0, 0.7), (6.0, -3.0), (11.0, 2.5), (-4.0, 5.0), (8.0, -8.0),
         (0.5, 0.2), (14.0, 1.0), (-9.0, -4.0), (3.0, 9.0), (5.5, -1.5)]


def _errors(display, live, H):
    truth = display.render()
    probe = profile_from_reference(truth, screen="b", display_size=display.size)
    rect = DisplayGeometry(H=H, display_size=display.size).rectify(live)
    return [
        (m.dx, m.dy)
        for spec in probe
        for m in [measure_translation(truth, rect, spec)]
        if m.dx is not None and m.zncc and m.zncc > 0.5
    ]


def run(*, bezel_grey=38, ratio=0.95, sensor=(2400, 1500), noise=1.2,
        defocus=0.9, theme="day", compare_board=False):
    absolute, relative, board, refused = [], [], [], 0
    for i, (tilt, roll) in enumerate(POSES):
        display = ClusterDisplay(theme=theme)
        display.state.update(NOMINAL)
        panel = BezelPanel(bezel_grey=bezel_grey)
        rig = BezelRig(display, panel, VirtualCamera(
            display_size=panel.panel_size, sensor_size=sensor,
            sampling_ratio=ratio, tilt_deg=tilt, roll_deg=roll, seed=i,
            noise_sigma=noise, defocus_sigma=defocus))
        undistort = Undistorter(Intrinsics(
            K=rig.camera.K, dist=rig.camera.dist, image_size=sensor))
        rig.show("main")
        live = undistort(rig.read())
        try:
            geometry = homography_from_display_aperture(
                live, display_size=display.size)
        except RuntimeError:
            refused += 1
            continue
        errors = _errors(display, live, geometry.H)
        if not errors:
            refused += 1
            continue
        arr = np.array(errors)
        absolute.append(float(np.sqrt(np.mean(np.sum(arr ** 2, axis=1)))))
        relative.append(float(np.sqrt(np.mean(
            np.sum((arr - arr.mean(axis=0)) ** 2, axis=1)))))
        if compare_board:
            rig.show("checkerboard")
            fitted = homography_from_display_pattern(
                undistort(rig.read()), PATTERN,
                chessboard_display_points(PATTERN, SQUARE_PX, ORIGIN),
                display_size=display.size)
            rig.show("main")
            board_errors = np.array(_errors(display, live, fitted.H))
            board.append(float(np.sqrt(np.mean(np.sum(board_errors ** 2, axis=1)))))
    return absolute, relative, board, refused


def _line(label, absolute, refused, relative=None):
    if not absolute:
        print(f"   {label:<30}        refused   [{refused}/{len(POSES)}]")
        return
    text = f"p50 {np.median(absolute):6.3f}  max {max(absolute):6.3f}"
    if relative is not None:
        text += f"    p50 {np.median(relative):6.3f}"
    tail = f"   [{refused}/{len(POSES)} refused]" if refused else ""
    print(f"   {label:<30}{text}{tail}")


def main() -> None:
    print(__doc__.strip().split("\n\n")[0])
    print()

    absolute, relative, board, refused = run(compare_board=True)
    print("against the route that needs the screen:")
    print(f"   {'':<30}{'absolute':>21}{'offset removed':>18}")
    _line("aperture (needs nothing)", absolute, refused, relative)
    _line("chessboard (needs the screen)", board, 0)
    print()

    print("sampling ratio (sensor sized so the whole panel fits):")
    for ratio, sensor in ((0.5, (2400, 1500)), (0.7, (2400, 1500)),
                          (0.95, (2400, 1500)), (1.3, (4000, 2600)),
                          (2.0, (6000, 3600)), (3.0, (8000, 5200))):
        a, _, _, r = run(ratio=ratio, sensor=sensor)
        _line(f"ratio {ratio}", a, r)
    print()

    print("contrast between panel and trim (the screen's background is ~13):")
    for grey in (18, 22, 25, 38, 100, 160):
        a, _, _, r = run(bezel_grey=grey)
        _line(f"trim grey {grey}", a, r)
    print()

    print("conditions:")
    for label, kwargs in (("night theme", {"theme": "night"}),
                          ("heavy noise", {"noise": 5.0}),
                          ("soft focus", {"defocus": 2.5}),
                          ("sharp, moire risk", {"defocus": 0.2})):
        a, _, _, r = run(**kwargs)
        _line(label, a, r)


if __name__ == "__main__":
    main()

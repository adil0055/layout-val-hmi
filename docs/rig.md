# Stage 0 — the rig, which decides everything downstream

This is the part that gets skipped, and then fought in software for six months.
Lock the following down before writing comparison code, and write the settings
into the calibration file (`Calibration.rig`) so a run record says what produced
it.

## Optics

**Sampling ratio.** You want at least 2 camera pixels per display pixel to
resolve a 1-display-pixel shift with sub-pixel interpolation; 3× is comfortable.
A 1080p camera aimed at a 1920-wide cluster samples at roughly 1:1, and the
honest floor there is about ±1 display pixel. Know which regime you are in before
promising anyone a tolerance.

`layoutval calibrate-geometry` reports the ratio and warns below 2:

```
sampling ratio: 1.57 camera px per display px
  NOTE: below 2 camera px per display px you cannot reliably resolve a
  1-display-pixel shift.
```

**Prime lens, locked.** Fixed focal length, focus and aperture mechanically
locked — grub screw or a dab of thread lock. A varifocal that creeps 0.2 mm
invalidates the calibration silently.

**Every automatic setting off.** Fixed exposure, gain and white balance.
Auto-exposure chases the content of the screen and changes edge profiles between
test cases, which later reads as an element that "moved". `CameraSource` turns
these off on open, but the properties are backend-dependent and may be ignored —
verify against your camera rather than trusting the return value, and record
`CameraSource.settings()` in the run.

**Global shutter** if anything on screen animates. A rolling shutter shears
moving content, and a sheared needle is not at the angle you measured.

## Two artefacts specific to photographing a display

**Backlight PWM banding.** Cluster backlights dim by pulse-width modulation,
typically a few hundred Hz to a couple of kHz. If exposure time is not a whole
multiple of the PWM period, brightness varies frame to frame and band position
drifts. Use a long exposure (≥ 10 ms), or measure the PWM frequency and lock
exposure to a multiple of it. Median-stacking several frames suppresses what is
left.

Not all of it, though — and the residue is proportional to brightness, so a large
bright element can differ *from itself* by more than a small telltale differs
from its own off state. That is why `TeachInSession` derives its threshold from
captures of the same state rather than from a constant
(`TeachInSession.adaptive_floor`). A fixed threshold here silently teaches the
wrong region, with a plausible box, correctly labelled with the signal you
toggled. It is in `tests/test_teachin.py`.

**Moiré.** The sensor grid beats against the display's pixel grid and generates
structure that template matching will happily lock onto. Fixes, in order of
preference: avoid a near-integer sampling ratio; capture well above display
resolution and downsample with a proper low-pass; or deliberately defocus by
roughly one display pixel. A slightly soft image with no moiré measures better
than a sharp one with it.

## Mounting and light

Rigid mount, ideally not sharing a bench with anything that moves. Enclose the
rig or use a polarising filter — the display is emissive, so ambient light
contributes nothing but reflections off the cover glass.

Allow 10–15 minutes of warm-up before a run: both the display and the camera body
expand as they heat, and that shows up as a slow drift of a pixel or two. This is
why the repeatability study spreads its frames over ~30 minutes rather than
taking them back to back — otherwise it measures sensor noise and misses the
drift that will be present in every real run.

## Drift, and why it must be noisy out loud

Rigs get bumped. `DriftTracker` runs `cv2.findTransformECC` with
`MOTION_EUCLIDEAN` against a region that never changes and applies the correction
on top of the base homography.

The correction is cheap. The important part is that it is **loud**: the magnitude
is logged on every run and a run that exceeds the alarm is flagged. Silent
compensation is how a rig that someone knocked last Tuesday keeps producing green
results for a month.

```python
est = tracker.measure(undistorted)
report.metadata["rig_drift_px"] = est.magnitude_px
if est.exceeds:
    report.flag("rig_drift", severity="review", drift_px=est.magnitude_px)
```

A non-converging ECC counts as exceeding: the static region no longer looks like
the static region, which is itself the finding.

## Before there is a rig at all

`layoutval capture-server` lets a phone stand in for the camera while the mount
is still being built — see the README. It is genuinely useful for shaking out
the geometry and the profile, and it is not a measurement rig:

- **The pose changes every shot.** Handled by re-solving each frame against the
  reference, which works for per-element faults and silently absorbs a fault
  where the whole layout moved. `--fixed-camera` turns the re-solve off once the
  phone is clamped, and then the pose change is reported instead of corrected.
- **Sampling ratio is usually poor.** A phone at arm's length covers a cluster
  at close to 1:1, where the honest floor is about ±1 display pixel. The
  calibrate step reports the ratio; believe it.
- **Exposure and white balance are automatic**, which is the thing the rest of
  this page says to turn off. Most phone cameras can be locked by holding on
  the subject; do it, and re-calibrate afterwards.

Everything below still applies to the rig you are heading towards.

## Getting the homography

Three ways, in order of preference.

**A. The cluster renders the pattern itself.** If the HMI build can be put into a
diagnostic mode showing a full-screen checkerboard, you get a direct, exact
correspondence between framebuffer and camera coordinates. This is the
highest-leverage thing to ask the HMI team for, and a one-off "layout calibration
screen" in test builds pays for itself permanently.
`homography_from_display_pattern`.

**B. ChArUco on the bezel.** ChArUco rather than plain ArUco specifically for
corner accuracy — the ArUco squares give identity and occlusion tolerance, but
the interpolated corners belong to a chessboard, and chessboard corners refine far
more accurately. One caveat straight from the OpenCV docs, and implemented in
`homography_from_charuco`: when the result feeds a homography, **disable** marker
corner refinement, because the proximity of the chessboard squares makes the
sub-pixel step deviate and those deviations propagate into the interpolated
corners. You also need a one-time measurement of where the active area sits
relative to the markers — do it once with method A and store it.

**C. The display's own edges.** Show full white, threshold, fit lines to the four
edges, intersect for corners. *Fit* lines — do not take corner points directly; a
line fit averages over hundreds of edge pixels and lands well under a pixel where
a corner detector lands at about one. `homography_from_display_edges`. Workable,
least stable, because it re-derives the geometry from content.

## Half-pixel conventions

Check them against synthetic ground truth once, and then you never have to think
about it again. Getting one wrong costs a constant sub-pixel offset on every
element, which looks exactly like a real systematic finding.

The simulator had this bug during development: supersampled rendering mapped
display coordinate `x` to subpixel `round(x*S)` instead of
`round(x*S + (S-1)/2)`, which put a fixed 0.375 px offset into every element.
It showed up as a "needle pivot bias" and was chased in the wrong module for a
while. `tests/test_calibration.py::test_rectified_frame_matches_the_framebuffer`
is what would have caught it.

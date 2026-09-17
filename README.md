# layoutval — camera-based layout validation for HMI clusters

Checks that every element on an instrument cluster sits where the reference says
it should, from a fixed camera, to a fraction of a display pixel.

The position this implements: **layout validation is a measurement problem, not a
detection problem.** The camera is fixed, the cluster is fixed, and the element
inventory is known before the test runs. Nothing needs to be *discovered* — it
needs to be *measured*, in display pixels, against a design reference.

The accuracy of the whole system is set by how well camera pixels map onto
display pixels, not by how cleverly elements are found in the image.

```
capture ──▶ undistort ──▶ rectify ──▶ locate ──▶ score ──▶ report
 frames      K, dist         H         Δx, Δy      tol      JSON
                    └── the accuracy budget lives here ──┘
```

## Quick start

```bash
pip install -e ".[dev]"
layoutval demo --out out/demo      # the whole pipeline against a simulated rig
pytest                             # 60+ tests, no hardware needed
```

`layoutval demo` walks the full build order against a synthetic cluster and a
virtual camera, injects defects with known ground truth, and checks that what
comes out is what was actually done to the display:

```
TELLTALE_ABS          injected (+3.00,+1.25) px  measured (+3.00,+1.00) px  -> FAIL/position
TELLTALE_BATTERY_LOW  injected wrong glyph       zncc=0.757                 -> FAIL/wrong_content
NEEDLE_SPEED          injected +4.00 deg         measured +4.09 deg         -> FAIL/angle
FUEL_BAR              injected (+2.25,+0.00) px  measured (+2.02,-0.01) px  -> REVIEW/position_marginal
```

## Commands

| Command | When |
|---|---|
| `layoutval calibrate-intrinsics` | Once per camera and lens. Gates on reprojection error. |
| `layoutval calibrate-geometry` | Once per rig build. Solves display→camera and reports the sampling ratio. |
| `layoutval import-design` | Builds a profile from an HMI design-tool export. |
| `layoutval repeatability` | Per-element measurement noise, σ, on a static screen. |
| `layoutval linearity` | Per-element sub-pixel bias. **A static screen cannot show you this.** |
| `layoutval gate` | Are these tolerances defensible on this rig? |
| `layoutval run` | Validate captured frames; JSON, overlay, heatmap, JUnit. |
| `layoutval capture-server` | Capture from a phone on the same network; measure here. |

## The parts

| Module | Does |
|---|---|
| `capture` | Frame bursts, median stacking, settling detection. |
| `calibration` | Intrinsics, three homography solvers, drift tracking with an alarm. |
| `profile` | The element inventory (`layout_profile.yaml`), design-export import. |
| `teachin` | Differential teach-in — toggle a CAN signal, diff, label by construction. |
| `measure` | ZNCC + phase correlation, mask centroids, needle pivot and angle. |
| `verdict` | Ordered per-element rules. |
| `residual` | SSIM residual (advisory) and explicit z-order assertions. |
| `repeatability` | σ per element, and the week-2 gate. |
| `linearity` | Sub-pixel bias per element. |
| `report` | JSON, annotated overlays, JUnit. |
| `server` | Phone capture over the local network, and the page it opens. |
| `authoring` | Snap assistance and a segmentation-model hook — **authoring only**. |
| `simulator` | Synthetic cluster and virtual camera, for tests and the demo. |

## Capturing from a phone

Before there is a mount and a lens, it is useful to point a phone at the screen
and have the answer come back. Start the server on the machine that will do the
measuring:

```bash
layoutval capture-server --profile profiles/main.yaml --pattern 14 5 --square-px 100
```

It prints a URL carrying a one-run token. Open it on a phone on the same
network and walk three steps:

1. **Calibrate** — the cluster shows its chessboard; this solves display-to-camera.
2. **Reference** — the cluster shows the screen under test, correct.
3. **Validate** — the same screen with whatever you are testing.

The verdict, the failing elements and the annotated overlay come back to the
phone. Captures, reports and overlays land in `--out`.

**A phone in your hand is not a fixed camera, and this does not pretend
otherwise.** Every number here rests on the camera not moving, and a hand-held
frame is at a different *pose*, not just a different position. So each frame's
pose is re-solved against the reference before measuring, and how far it had to
go is reported. That buys back a usable measurement at a cost worth stating
plainly: **a correction that re-solves the whole pose also absorbs a fault in
which every element moved together.** Per-element faults survive it — the rest
of the frame dominates the fit — but a whole-layout shift does not. Clamp the
phone and pass `--fixed-camera`, and the pose is checked rather than re-solved.

Two more things a phone brings with it. Its photos carry an EXIF orientation
rather than rotated pixels, which is handled — a frame that came in on its side
would calibrate and measure perfectly happily and be wrong about everything.
And at arm's length a phone is often under-sampled; the calibrate step reports
the sampling ratio and says so below 2.

It binds to the local network and accepts uploads, so it is a bench tool: every
URL carries a token minted at startup, uploads are capped and must decode as an
image, and nothing from an upload is executed or used as a path. Stop it when
you are done.

## Three things worth knowing before using it

**Work in display space.** After rectification every position, tolerance and
error is in display pixels. That makes a tolerance something the HMI team can
argue about ("the icon may not move more than 2 px") rather than something only
the rig owner can interpret.

**Two metrics, not one.** Phase correlation and ZNCC are both computed for every
element. ZNCC answers *is this still the same thing*; the shift estimate answers
*how far did it move*. Running only one is how an element rendering the wrong
symbol gets reported as a position failure and someone loses a day in layout
code. When the two shift estimates disagree by more than the coarse stage's own
precision, the record says so and says which one was used.

**Learned models in the authoring loop, deterministic code in the measurement
loop.** Not an anti-ML position — it puts ML where its failure mode is "an
engineer adjusts a box" rather than "a safety telltale defect ships".
`tests/test_architecture.py` enforces the separation.

## Build order

| Week | Work | Deliverable |
|---|---|---|
| 1 | Rig lock-down, intrinsics, undistortion | Stored calibration, reprojection error < 0.3 px |
| 2 | Display-space rectification, **repeatability + linearity studies** | Per-element σ and bias — **the gate; don't build past it** |
| 3 | Differential teach-in over the full inventory | `layout_profile.yaml` with templates |
| 4 | Measurement core | Per-element records with sub-pixel deltas |
| 5 | Verdicts, tolerances, residual, reporting | JSON + annotated overlay |
| 6 | Variant coverage, CI, flake hunt | Suite green across themes and display variants |

Week 2 is a gate on purpose. Everything after it is straightforward engineering;
everything before it decides whether those weeks' numbers mean anything.

## Where this departs from the usual advice

Both departures are measured, not asserted. See
[docs/method.md](docs/method.md) and `benchmarks/estimator_comparison.py`.

1. **Sub-pixel ZNCC beats phase correlation on cluster content**, by 2–3× in
   RMS error against a known injected shift, consistently across noise levels,
   sampling ratios and focus. Cluster elements are sparse high-contrast glyphs on
   a near-uniform background, which is not the richly textured patch phase
   correlation is good at. Both are computed; the default reports ZNCC and
   cross-checks with phase; the linearity study can pick per element.

2. **A repeatability study is not enough to set a tolerance.** It measures noise
   on a static screen. Correlation estimators also suffer peak locking — a bias
   that depends on the *fractional* part of the displacement and is therefore
   exactly zero when nothing moves. In this implementation's own simulator one
   element measured σ = 0.005 px and 0.28 px of systematic error in the same run.
   `tol_fail ≥ 3σ + spec` would have been about twenty times too tight. The
   defensible floor is `3σ + subpixel_bias + spec_tolerance`.

## Documentation

- [docs/rig.md](docs/rig.md) — stage 0: optics, PWM banding, moiré, mounting
- [docs/method.md](docs/method.md) — what was measured, and how to re-run it
- [docs/traps.md](docs/traps.md) — animations, themes, needles, z-order, drift
- [docs/licences.md](docs/licences.md) — every dependency, checked

## Licence

Apache-2.0. Every runtime dependency is Apache-2.0, BSD or MIT — no AGPL, no
non-commercial weights. See [docs/licences.md](docs/licences.md).

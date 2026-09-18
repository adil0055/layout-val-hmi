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
| `autoprofile` | An inventory taken from the reference frame, when there is no authored one. |
| `authoring` | Snap assistance and a segmentation-model hook — **authoring only**. |
| `simulator` | Synthetic cluster and virtual camera, for tests and the demo. |

## Capturing from a phone

Before there is a mount and a lens, it is useful to point a phone at the screen
and have the answer come back. Start the server on the machine that will do the
measuring:

```bash
# with no pattern at all: hand it the framebuffer and it matches the screen
./run.sh --screenshot screen.png        # on the cluster machine
layoutval capture-server --render screen.png

# or with the cluster's chessboard, which is the more accurate of the two
./run.sh --calibration checker --calibration-export board.json
layoutval capture-server --board board.json --profile profiles/main.yaml
```

`--render` takes a display-space image of what the cluster is drawing and
calibrates by matching the screen's own content, so the cluster never has to
leave the screen under test. **It needs the framebuffer, so it is for a
simulated or developer-controlled HMI** — a production cluster gives you a
camera and nothing else, and then the geometry has to come from a bezel marker
(`--charuco`, below), one bright frame, or marked corners; see
[What a real cluster leaves you](docs/rig.md#what-a-real-cluster-leaves-you). It measures within about 0.02 px of the chessboard
across pose, focus, sampling ratio and lens distortion — see
`benchmarks/calibration_methods.py` and [docs/rig.md](docs/rig.md). The content
does not have to match the photograph exactly; RANSAC discards whatever moved.

`--aperture` calibrates from the display's own physical border and asks the
cluster for **nothing at all** — no pattern, no framebuffer, not even the one
binding frame `--charuco` needs. It is the route for a real cluster, and it is
not a fallback: measured over ten poses it lands at 0.052 px against the
on-screen chessboard's 0.072 px, because fitting four lines over the whole
display boundary averages thousands of edge pixels where a chessboard localises
each corner on its own.

It needs `--intrinsics`, and needs them harder than any other route: it fits
straight lines to the display's edges, and lens distortion bows exactly those
lines. Measured, 0.04–0.06 px undistorted on any lens against 1.3 px on a mild
one and 4–8 px on a normal phone lens. Solve them once from the phone — pick
**Intrinsics** on the capture page and shoot 12 views of a board
(`--lens-board`), which never goes near the cluster because it is the camera
being measured: print it, or open the PNG on any other screen.

`9x6:30:22` measured best (k1 error 0.0009 against a true −0.09), but the
choice is not delicate — 7x5 and 16x3 both land inside 0.007, and what actually
matters is filling the frame and varying the pose between shots. Eight views is
already enough; the twelve it asks for are slack. A solve that goes wrong goes
*visibly* wrong: the worst board tested reported 0.374 px reprojection error,
above the 0.3 px this package gates on, so it is flagged rather than believed.

It locates the physical opening, so display coordinates from it carry a
constant offset against the active area behind the mask. That cancels exactly
between reference and validate, so defect measurements are unaffected; pass
`--display-inset X Y` only if you need absolute coordinates. It needs the whole
display plus a margin of trim in frame, and enough light to tell panel from
trim — below about ten grey levels of contrast it refuses rather than guesses.
See `benchmarks/aperture_calibration.py` and [docs/rig.md](docs/rig.md).

`--charuco 16x3:30:22` is the route for a cluster you cannot ask to draw
anything: a ChArUco board stuck to the bezel, outside the active area.
`layoutval charuco-board 16x3:30:22` draws the board to print (lengths are
millimetres; print at 100%, and matt, because a glossy board under a cluster's
own glass gives you two reflections to fight). The first
Calibrate **binds** it — that one frame needs the markers *and* the calibration
screen together — and every Calibrate after needs the markers only, so the
screen stays on the screen under test. The binding is saved and picked up by
later runs.

It needs `--intrinsics` and refuses without them: the markers are photographed
away from the screen's part of the frame, so lens distortion does not cancel the
way it nearly does for a board on the screen. Measured, 0.10 px undistorted on
any lens against 0.6–5.7 px with the distortion left in. Re-bind if the camera
moves much — reusing an anchor across a move costs about 5× (0.09 px → 0.25–0.32
px, 0.94 px at p95), and the server says when the board has drifted from where
it was bound. `benchmarks/bezel_anchor.py` has the full table, and
[docs/rig.md](docs/rig.md) the caveat about a recessed display that the
simulator cannot settle.

`--board` takes the file the cluster's own `--calibration-export` writes, which
carries the exact corner coordinates. Prefer it over describing the board with
`--pattern`/`--square-px`/`--origin`: nothing has to be guessed, and it uses the
pixel-centre corners rather than Qt's, which differ by half a pixel — and that
half pixel goes straight into the homography and from there into every
measurement taken through it.

It prints a QR code and a URL, both carrying a one-run token. **Scan the code**
— nobody should be hand-typing a token into a phone, and the first version of
this made people do exactly that. If you do type it, the token is
case-insensitive and drawn from an alphabet with no `0`/`O` and no `1`/`l`/`I`,
because those are the characters that get mistyped.

A link that is refused says why on the phone: mistyped, server restarted since
(the token changes every run), or the `?t=…` lost off the end.

Open it on a phone on the same network and walk three steps:

1. **Calibrate** — the cluster shows its chessboard; this solves display-to-camera.
2. **Reference** — the cluster shows the screen under test, correct.
3. **Validate** — the same screen with whatever you are testing.

The verdict, the failing elements and the annotated overlay come back to the
phone. Captures, reports and overlays land in `--out`.

### Without a layout profile

There is nothing to measure until an inventory exists, and building one —
exporting it from the design tool, or teaching it by toggling CAN signals — is a
piece of work. So when no `--profile` is given, the **Reference** step takes an
inventory from the reference frame itself: a cluster is bright elements on a
dark background, which segments cleanly, and each lit region becomes a
measurable element.

It works, to the same sub-pixel accuracy, and it is the weaker of the two
questions:

- it answers **does this frame match the reference frame**, not *does the build
  match the design*. A layout error present when the reference was taken is
  baked into the reference and will never be reported;
- it cannot name anything, so a defect comes back as `auto@312,75` rather than
  `TELLTALE_ABS` — the annotated overlay is what turns that back into an
  element;
- it treats everything as fixed, so the cluster has to stay in the same state.
  A needle at a different speed is a moving element to an authored profile and a
  failure to this one.

Pass `--no-auto-profile` to refuse rather than measure, and `--profile` once
there is a real inventory; an authored profile is never replaced by a discovered
one.

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

- [docs/architecture.md](docs/architecture.md) — how the whole thing fits
  together, and why
- [docs/rig.md](docs/rig.md) — stage 0: optics, PWM banding, moiré, mounting
- [docs/method.md](docs/method.md) — what was measured, and how to re-run it
- [docs/traps.md](docs/traps.md) — animations, themes, needles, z-order, drift
- [docs/licences.md](docs/licences.md) — every dependency, checked

## Licence

Apache-2.0. Every runtime dependency is Apache-2.0, BSD or MIT — no AGPL, no
non-commercial weights. See [docs/licences.md](docs/licences.md).

# Architecture

How this package is put together, and why it is put together that way. For what
was *measured* rather than designed, see [method.md](method.md); for the rig
that decides the accuracy budget, [rig.md](rig.md); for the failure modes this
domain is full of, [traps.md](traps.md).

---

## 1. The problem, stated precisely

Validating an instrument cluster's layout from a camera is a **measurement**
problem, not a detection problem.

That distinction drives everything below. A detector answers "is there a
battery telltale in this frame?" — which is not the question. The question is
"is the battery telltale within 2 px of where the design says it should be, and
is it still the battery telltale?" The first question is answered by a model
with a confidence score; the second is answered by a number with an error bar.

Consequences that fall straight out of it:

- **A number needs units.** Everything geometric is reported in *display
  pixels*, never camera pixels. A tolerance in display pixels is something the
  HMI team can argue about ("the icon may not move more than 2 px"). A
  tolerance in camera pixels is something only the rig owner can interpret, and
  it silently changes meaning when the camera moves.
- **A number needs a known error.** A tolerance that is not backed by a
  repeatability study is a guess. `Tolerance.defensible_floor()` makes the
  arithmetic explicit and refuses to let it stay implicit.
- **A number needs to say when it is not a number.** Every stage that cannot
  measure something raises or records an error rather than returning a
  plausible value. This is the single most repeated decision in the codebase.

---

## 2. Two invariants

### 2.1 Work in display space

The pipeline rectifies every frame into the framebuffer's own coordinate system
before measuring anything. `capture -> undistort -> rectify` happens once, and
from `rectify` onwards nothing knows or cares where the camera was.

This is what makes results comparable across rigs, across sessions, and against
a design export. It also concentrates the entire geometric accuracy budget into
one place — the homography — which is why [rig.md](rig.md) is as long as it is
and why there are five different ways to obtain it (§5).

### 2.2 Two metrics, never one

Every correlation-based measurement produces **both**:

- a **displacement** — how far it moved;
- a **ZNCC identity score** — whether it is still the same thing.

Running only displacement is how an element rendering the *wrong symbol* gets
reported as a position failure, sending whoever picks up the defect into layout
code for a day. `verdict.py` checks identity *before* position for exactly this
reason (§7).

---

## 3. The architectural line: learned models in the authoring loop, deterministic code in the measurement loop

This is the load-bearing structural decision.

`layoutval.authoring` may use a segmentation model to *propose* element boxes
during inventory building. Nothing in the measurement path may import a learned
model, or import `authoring`.

It is not an anti-ML position. It is about where a failure lands:

| | failure mode |
|---|---|
| model in the authoring loop | an engineer adjusts a box |
| model in the measurement loop | a safety telltale defect ships |

A model in the measurement loop also destroys traceability: you can no longer
say *why* a frame passed, only that a network said so. And it makes the result
non-reproducible across a weights update.

**This is enforced by a test, not by convention.** `tests/test_architecture.py`
parses the AST of every module on the measurement path and asserts that none of
them imports `torch`, `tensorflow`, `ultralytics`, `transformers`, `sklearn`,
`onnxruntime`, … or `layoutval.authoring`. It also asserts that `authoring`
declares its backend as a `Protocol`, so the package never depends on a model at
import time, and that no non-permissively-licensed dependency has crept into
`pyproject.toml`.

The measurement path is declared explicitly in that test:

```
measure.py  verdict.py  pipeline.py  calibration.py  capture.py
residual.py  repeatability.py  linearity.py  types.py  profile.py
```

Adding a module to the measurement path means adding it to that list.

---

## 4. The six stages

```
     ┌─────────┐   ┌───────────┐   ┌─────────┐
     │ capture │ → │ undistort │ → │ rectify │ ──────┐
     └─────────┘   └───────────┘   └─────────┘       │
       stage 1        stage 2        stage 3         │  display pixels
                                                     │  from here on
     ┌────────┐   ┌───────┐   ┌────────┐             │
     │ locate │ → │ score │ → │ report │ ←───────────┘
     └────────┘   └───────┘   └────────┘
      stage 4      stage 5     stage 6
```

Wired together in `pipeline.Pipeline`. `run()` acquires a frame, aborts if
acquisition flagged anything fatal — *without manufacturing numbers for it* —
and otherwise calls `measure_frame()`.

| stage | module | job |
|---|---|---|
| 0 | — | the rig itself ([rig.md](rig.md)) |
| 1 capture | `capture.py` | get frames; refuse frames not worth measuring |
| 2 undistort | `calibration.py` | apply lens intrinsics |
| 3 rectify | `calibration.py` | homography into display space |
| 4 locate | `measure.py` | where did each element actually land |
| 5 score | `verdict.py` | ordered rules → verdict + reason |
| 6 report | `report.py` | structured JSON + annotated overlay |
| — | `residual.py` | what you did not model (advisory) |

**Stage 1 does two jobs**, and the second is the interesting one. Median-stacking
N frames kills sensor noise and backlight PWM ripple for free. The *settling
detector* exists because needles sweep, popups slide and telltales fade:
measuring during a transition produces a number that is precise, wrong, and
indistinguishable from a real defect.

---

## 5. Calibration: six routes to one homography

`calibration.py` is the largest module because stage 3 decides everything
downstream. All six routes produce a `DisplayGeometry` carrying `H`, the
`display_size`, a `method` string and a fit `residual_px`, so the report always
records *how* the frame was rectified.

| | route | needs from the cluster | when |
|---|---|---|---|
| **A** | `homography_from_display_pattern` | draws a chessboard | best case; ask the HMI team for a calibration screen |
| **B** | `homography_from_charuco` | one binding frame, then nothing | board stuck to the bezel |
| **C** | `homography_from_display_edges` | one full-white frame | fallback |
| **D** | `homography_from_screen_content` | its framebuffer | dev/simulated HMI only |
| **E** | `homography_from_display_aperture` | **nothing at all** | production cluster, clean scene |
| **F** | `homography_from_marked_corners` | **nothing at all** | any scene; the default for `layoutval go` |

**Route E is the one that matters for a real cluster**, which will not draw a
chessboard, will not hand over its framebuffer, and will not hold still on a
calibration screen. It locates the display's *physical opening* in the trim,
which is hardware and is present in every frame regardless of what the software
is doing. Measured, it is not a fallback: 0.052 px against route A's 0.072 px,
because fitting four lines along the whole display boundary averages thousands
of edge pixels where a chessboard localises each corner independently.

Route E's internals are worth knowing because three obvious implementations are
wrong, each measured:

- **Otsu does not work.** A cluster photograph has at least three populations —
  dark screen, mid trim, something bright — and a two-class split lands between
  the bright thing and everything else. It put the threshold at 123 and merged
  screen and trim into one blob covering 99.9% of the frame. The threshold is
  *swept* instead, and candidate regions are scored on stability across levels,
  rectangularity, and the framebuffer's aspect ratio (the one thing known for
  free).
- **The aperture is a hole.** `RETR_EXTERNAL` discards holes by definition, and
  taken the other way round the dark screen merges with the dark room behind the
  cluster. `RETR_LIST` was the difference between finding the border in every
  pose and in none.
- **A coarse answer is worse than none.** Falling back to threshold corners when
  sub-pixel refinement fails produced 1.7–3.1 px errors while every refined
  frame was inside 0.05 px — and nothing in the result distinguished them. It
  refuses now.
- **Nested rectangles are refused, not resolved.** An HMI in a window inside a
  monitor presents two rectangles of the same proportions. "Innermost wins" is
  correct in principle and was measured insufficient (5 of 14 arrangements),
  with the other nine silently wrong. Detecting the ambiguity and refusing took
  it to zero silently wrong, and the ambiguity is trivially removable by
  whoever is holding the camera.

Route E also carries a **constant offset**: it finds the physical opening, and
the active area sits behind a mask it cannot see. That offset *cancels exactly*
between reference and validate, since both rectify through the same homography,
so it does not affect a defect measurement. It matters only for absolute
comparison against a design, where `inset_px` takes the mask width.

### Route F, and proposing its corners

Route F takes four rough corner points and snaps each side to the real panel
edge sub-pixel. Where the points come from is a separate question with three
answers, and the phone can switch between them at any time (§11): a person taps
them, `displayfind.py` proposes them and a person confirms, or route A replaces
the lot when the cluster can draw its board.

`propose_display_corners` exists because the hard part of route E on a bench
photograph is not accuracy but *which rectangle*: a laptop on a desk offers the
screen, the window's title bar, the desktop's top bar, the lid and the keyboard,
all with long straight edges. What it does:

- **Line segments, not contours.** LSD on a CLAHE-equalised copy (the
  equalisation took a fragmented bottom edge after an exposure change from
  failing to 120/120), grouped into near-horizontal and near-vertical families
  and merged collinearly.
- **Each side is chosen on its own, innermost first, by one rule: nothing lit
  sits on a bezel.** For a candidate side, lit marks (a white top-hat) are
  counted in a band just beyond it -- 60% of the way to the next line outward,
  so the band stays on the bezel and does not reach the keyboard or the desk --
  and only between the two neighbouring sides, sampled at every pixel. A title
  bar fails it (the clock and the window title are beyond it); the panel's edge
  passes it. Scoring whole quadrilaterals was tried first and kept choosing lid
  and keyboard; the brightness-polarity prior it leaned on was measured false on
  the bench photographs.
- **A side must cover the span between its neighbours** (at least 45%), so a
  short stray segment cannot stand in for an edge.
- **The winner is refitted at full resolution**, allowed to move at most 0.5% of
  the diagonal, and the proposal says whether it is `confident`.

Measured on the two bench photographs under 120 random warps -- perspective,
rotation, exposure -- it found the screen every time, and was never confidently
wrong. It proposes on the undistorted frame and maps the dots back into the
photograph's own pixels: proposed on the raw photograph instead, lens distortion
bowed the edges and the corners landed 29-34 px off, far enough for route F's
snap to refuse them. Accepted unchanged, the proposal now snaps to within
0.03 px of route E's border fit. It is deterministic line geometry and sits on
the measurement path under the same no-learned-models test as everything else.

### Drift

`DriftTracker` re-solves pose per frame against the reference. In
`MOTION_EUCLIDEAN` mode it corrects a nudge; in homography mode it absorbs a
full pose change, which is what makes a hand-held phone usable at all. The cost
is stated rather than hidden: a correction that re-solves the whole pose also
absorbs a fault in which *everything* moved together. Per-element faults survive
it because the rest of the frame dominates the fit; a whole-layout shift does
not. `--fixed-camera` checks pose without re-solving.

---

## 6. Measurement (stage 4)

### Element kinds decide the estimator

`ElementKind` is not cosmetic — picking the wrong kind is the most common way to
get a confidently wrong number out of this pipeline.

| kind | estimator | why |
|---|---|---|
| `ICON` | sub-pixel ZNCC + phase correlation | shift-invariant patch |
| `TEXT` | same, but always on luma | sub-pixel anti-aliasing puts colour fringes on glyph edges whose weight depends on sub-pixel phase |
| `TELLTALE` | mask centroid + ZNCC identity | isolated luminous element on a dark background |
| `NEEDLE` | pivot + angle, as two separate quantities | rotates about a pivot; not a shift-invariant patch, so phase correlation returns nonsense |
| `REGION` | correlation; also the fallback | generic textured widget |

A needle at the right angle about the wrong pivot is a real and separate defect,
which is why `measure_needle` reports both.

### Search is always bounded

Correlation is confined to the expected box dilated by the element's margin,
never the whole screen — which is how you match a similar-looking element
somewhere else entirely. If the correlation peak lands *on the border* of the
search window, the element may have moved further than the window can measure:
that is recorded as `peak_on_search_border` and treated as a failure of unknown
magnitude, never as a clamped value.

### Guards

- `is_degenerate()` — a uniform patch correlates perfectly with anything.
- `element_absent` — "not drawn at all" and "drawn wrongly" are different
  defects that go to different people.
- `estimator_disagreement_px` — ZNCC and phase correlation are both computed;
  when they disagree materially, that is information worth carrying.

---

## 7. Verdicts (stage 5)

`verdict_for()` applies **ordered** rules. The ordering is the design:

1. `element_absent` → **FAIL** (absent)
2. measurement error → **FAIL**
3. `zncc < identity_min` → **FAIL** (wrong content)
4. `peak_on_search_border` → **FAIL** (missing or displaced)
5. `|delta| > tol_fail` → **FAIL** (position)
6. angle beyond `angle_fail_deg` / `angle_warn_deg` → **FAIL** / **REVIEW**
7. `|delta| > tol_warn` → **REVIEW** (position marginal)
8. otherwise → **PASS**

Identity before position, absence before both. Each verdict carries a `reason`
constant and `explain()` renders one line a defect report can carry verbatim.

---

## 8. Tolerances, and what makes one defensible

```
defensible_floor() = 3σ + subpixel_bias + spec_tolerance
```

- **σ** — measured 1-sigma repeatability, from `repeatability.py`.
- **subpixel_bias** — worst systematic error as a function of where between two
  pixels the element lands, from `linearity.py`.
- **spec_tolerance** — what the requirement actually asks for.

`check_against_sigma()` complains, in words, when `tol_fail` sits below that
floor. A rig that cannot resolve the requirement should be reported and fixed,
not shipped: **a suite that fails randomly gets marked flaky and ignored inside
two months, which is worse than not having the test.**

The middle term is the one people drop, and it is often the largest of the
three. A static-screen repeatability study cannot see it — that is a measured
finding, not a theoretical worry (see [method.md](method.md) §2: σ = 0.005 px
from a static study against 0.28 px of real error).

---

## 9. The element inventory

The pipeline only *measures*; the inventory is established beforehand.
`ElementSpec.source` records where each expectation came from, and it is
carried into the report because **it decides what a green result means**:

| source | module | a green result means |
|---|---|---|
| `design` | `profile.py` (design export) | the build matches the design |
| `teachin` | `teachin.py` | the build matches the last build |
| `manual` | `authoring.py` | a human drew a box |
| auto | `autoprofile.py` | this frame matches the reference frame |

**Teach-in** is the workhorse, and its trick is that the label is correct *by
construction*: toggle a known CAN signal, diff two median-stacked captures, and
whatever changed is that signal's element. No annotation pass, no labelling
errors, no drift between what the design tool calls an element and what the CAN
matrix calls it. For elements that cannot be toggled, drive the underlying value
across its range — what moves is the needle or the digits, what stays is the
static plate, and that separation is useful on its own because the two need
different validation.

**Auto-profile** is the fallback for when no inventory exists yet. A cluster is
bright elements on a dark background, which segments cleanly (Otsu + connected
components), so every lit region becomes an element whether or not anyone has
named it. It buys sub-pixel accuracy equal to an authored profile against the
question "does this frame match the reference frame" — and it explicitly does
*not* answer "does the build match the design".

It leaves out what could be segmented but not measured. A region whose detail
is one-dimensional -- a straight line or edge, scored by the structure tensor's
smaller eigenvalue over its larger, the Shi-Tomasi corner test -- looks the same
all along its length, so its position along it cannot be found: compared with
itself, a 370 px edge on a bench photograph matched 8 px down its own length and
came back FAIL. On two bench photographs lines scored under 0.02, a long bar
0.07, and every tick, digit and icon 0.12 or more; the cut is 0.05. A region
touching the part of the rectified frame the camera never saw -- corners placed
past the photograph's edge leave empty border -- is the photograph's edge, not
an element. Neither kind was ever measurable; both used to be FAIL.

---

## 10. The residual check

Per-element checks only find problems with elements you knew about. After
measuring everything on the list, `residual.py` diffs the rectified frame
against the reference and looks at what is left over — the stray artefact, the
element nobody taught, the wrong z-order.

**Advisory, always.** Pixel-level comparison of camera captures is noisy enough
that a hard threshold either fires constantly or is set so loose it catches
nothing. Its job is to surface things a per-element measurement passes cleanly,
for a human to look at. A finding makes a run with every element passing
REVIEW, and the phone says so in words and boxes it on the overlay.

**It has to be a real change.** SSIM compares texture relative to contrast, so
on a dark flat part of a screen it scores the faintest disagreement as total:
the moire of the panel's pixel grid, the noise left where a reflection was
subtracted, what remains of a smear once two photos are blurred to match. Two
photographs of one screen came back "80 pass, 0 review, 0 fail" under the word
REVIEW for it. A region is now a finding only if the brightness there changed
by 40 grey levels or more (the 90th percentile of the difference); something
drawn or erased changes it by a hundred or more, those by a few to a few tens.

---

## 11. The capture server

`server.py` is a small stdlib HTTP server plus an embedded single-page capture
UI, so a phone on the same network can shoot frames that are measured on the
laptop. It exists because a rig takes a mount, a lens and an afternoon, and long
before that it is useful to point a phone at the screen and get an answer.

`CaptureSession` holds all state behind one lock — uploads arrive on whatever
thread the server hands them to, and re-calibrating replaces state the next
request reads. Without the lock, two phones (or one impatient phone) could
measure against a half-replaced reference and the result would be *plausible*
rather than obviously wrong.

Actions, in the order a rig needs them:

| action | what it does |
|---|---|
| `intrinsics` | accumulate views of a lens board; solve when there are enough |
| `propose` | find the display's corners and send them back as dots; changes nothing |
| `corners` | calibrate from four corners, tapped or confirmed (route F) |
| `calibrate` | solve display-to-camera from the cluster's chessboard (route A) |
| `rebind` | re-tie a bezel board to the active area after the camera moves |
| `reference` | rectify and keep as the golden reference |
| `validate` | measure against that reference and answer |

**Modes.** `POST /mode` switches between `auto` (proposed corners), `manual`
(tapped corners) and `chessboard`, live. The corner modes map into the screen's
full resolution and the chessboard into the board's canvas, so a switch is a
different display space: it starts calibration over, drops the reference and a
discovered inventory, and keeps the lens solve, which belongs to the camera.
Choosing the mode already on is not a switch. (Left calibrated across a switch,
the page saw no reference, moved straight on to Reference, and the chessboard
photograph meant to calibrate became the reference.) The chessboard mode needs
the exact corners the cluster drew. With `--board` they come from the HMI's
export; without it, `layoutval go` carries the HMI's own layout rule
(`hmi_board`: 100 px squares, one square of quiet zone, centred, origin on a
whole pixel) for each skin's canvas, and the grid the camera finds says which
skin it is -- 14x5 inner corners on 1790x870, 16x4 on 1920x720, neither of which
fits inside the other. Checked against the HMI's own `pattern_geometry`, corner
for corner, and measured at 0.02-0.05 px against the simulator's true mapping.
With neither, the mode is refused: guessing a board is how a smaller grid
inside a bigger one solves at the wrong scale.

**Glare.** Every reference/test pair is de-glared before it is compared
(`glare.py`). A reflection off the cover glass is light added to what the display
emits, and a single photograph cannot say which is which -- but it can say what
is *large*: a reflection is a smooth hill or a flat-sided window, the artwork is
strokes. So:

- **Estimate** the large-scale light with a morphological opening (the
  rolling-ball background of shading correction), a square a tenth of the
  frame's short side. It keeps a window frame's hard edges and square corners
  exactly. A median filter was tried first; it rounded the corners and, once
  smoothed, softened the edges, leaving rims 50-70 levels high straight through
  the elements they crossed. For a day theme (dark artwork on light), a closing
  instead; the polarity is decided once, on the reference.
- **Subtract pairwise, in linear light.** Each frame loses the large-scale light
  the other lacks, and both are left on the dimmer of the two. Light adds in
  linear units, so that is where it comes off; in encoded values a reflection
  lifts black a long way and white hardly at all. One floor per frame was tried
  first and could not bring back down anything darker than the floor. A
  reflection in the same place in both frames stays in both, where it cancels.
- **Align on de-glared frames too.** The hand-held pose re-solve matches
  brightness, and with the camera perfectly still a moved reflection was read
  as 0.5-11 px of camera motion, dragging every element with it.
- **Re-check identity on detail where a reflection lay.** A compact lamp is
  only partly subtracted -- the opening cannot follow the top of a hill much
  narrower than its square -- and leaves a slope of light, which correlation
  reads as different content. An element that fails its identity check under a
  reflection is compared again at the measured position on its fine detail
  alone (each patch less a blur of itself) and without the pixels the
  reflection clipped. Over a good screen that turned a FAIL into a PASS; a 2 px
  fault under the same lamp still measured 2.03 px, and a wrong symbol under it
  still failed.
- **Never a verdict of its own.** Glare is the room, not the display. It is
  subtracted and the elements are measured; it does not put REVIEW on anything.
  (An earlier version did, for anything a reflection clipped and for every
  result against a reference with a strong reflection on it; on a glared bench
  photograph compared with itself that was 75 REVIEWs, and the rule was wrong
  for what the rig is for.) Residual-check findings that sit on a subtracted
  reflection become a note: the reflection's photon noise stays behind after its
  light is removed. How much light was taken off is a note in the saved report.
- **What is left.** An element a reflection turned completely white carries
  nothing; the camera recorded no content there, and its result is whatever
  measuring that white patch gives -- usually wrong content. The fix is to move
  the camera or shade the screen.

Measured through the whole capture session, hand-held, reflections moving
between the two shots: without it, four FAILs and two REVIEWs on a good screen
across six glare scenes; with it, a PASS in every scene where nothing clipped,
and a 2 px fault measured 1.70-1.78 px against 1.77 px with no glare. On the two
bench photographs, validated against themselves and against copies with
reflections added, it changes nothing that was right and passes everything that
was not. A 5-megapixel pair costs 0.2-0.5 s. Learned single-image reflection
removal was considered and rejected for the measurement path: it returns a
plausible image, and a measurement of a plausible image is a measurement of the
model. `--keep-glare` turns all of this off.

**Camera shake.** Moving the phone *between* shots is the pose re-solve's job;
blur *within* a shot is `blur.py`'s. The two aligned, de-glared frames are
related, to a good approximation, by one convolution, and with the sharp one
known the kernel is a regularised Fourier division (a Wiener estimate), cut to
a 20 px radius, cleaned of negative lobes and isolated specks, and normalised.
Both directions are tried, since either photo can be the shaken one; the
kernel is applied to the sharper photo only if it brings the two at least 10%
closer, and only after being re-centred on its own centroid, so it changes how
sharp an element looks and never where it is. It is one kernel for the whole
frame, fitted to every element at once, so an element that moved does not move
it. Measured: a 6 px streak had put up to 19 elements of a good screen at FAIL
and a 12 px one most of them; both now pass on both bench photographs, and a
2 px fault under a 12 px streak measured 1.56-1.77 px against 1.77 unshaken. At
20 px the shake is no longer one kernel across the frame -- perspective scales it
differently from one side of the display to the other -- and a few elements
read 1.5-2.7 px off.

**Full-resolution phone photographs.** A current iPhone shoots 24 MP. Two
steps were sized for smaller frames and, at 24 MP, ran long enough for the
phone to give up on the request, which Safari reports only as "Load failed":
the hand-held pose re-solve took 31-46 s and 3.4 GB whenever the phone had
moved between shots, and the chessboard search 12-18 s. The pose is now solved
on a copy no longer than 2000 px and conjugated back to full resolution
(3-4.5 s, under 1 GB; a 2 px fault measured 1.76-1.79 px against 1.82-1.85 at
full size). The chessboard is now found on a copy that size and each corner
refined on the full photograph, with a window scaled to the square: 0.7-0.9 s,
and more accurate, not less, since the fixed 5 px window it used before was
too small for corners that large (0.08-0.19 px rms against 0.38-0.88). The
measurement itself still rectifies from the full photograph. The terminal
prints a line as each photograph arrives, so a request the phone gave up on
still shows where it got to.

Design notes worth knowing:

- **Intrinsics are collected as correspondences, not frames.** A phone
  photograph is tens of megabytes; its correspondences are a few kilobytes.
- **The lens solve is gated on generalisation, not on fit.** Reprojection error
  measures how well the model fits the views it was given; a set of
  near-identical views is fitted beautifully and is wrong everywhere else, and
  reports a *lower* rms for doing so. Measured, a solve at 0.051 px rms left
  1.276 px of real error behind, and one that would have passed a 0.3 px gate
  was worse than skipping undistortion altogether. `LensCheck` therefore scores
  held-out reprojection and the spread of board angles, and a solve that fails
  is not adopted.
- **The lens board is independent of the calibration route** (`--lens-board`).
  Intrinsics belong to the *camera*, so tying them to a rig route would mean the
  one route that asks the cluster for nothing could only get undistortion by
  configuring a route that asks it for something.
- **Security is bench-grade and stated as such.** Binds to the local network,
  token in every URL, uploads capped and required to decode as an image, nothing
  from an upload ever executed or used as a path. Plain HTTP, because a phone
  camera over HTTPS needs a certificate the phone trusts; the page therefore
  uses the file-upload control that hands off to the phone's own camera app,
  which needs no secure context.
- **The token alphabet excludes confusables** (`0`/`O`, `1`/`l`/`i`) and is
  compared case-insensitively, because somebody will read it off a laptop and
  type it into a phone. The URL is also rendered as a terminal QR code.

---

## 12. The simulator

`simulator.py` is a **test fixture, not a model of any camera**. It exists so
the pipeline can be developed and demonstrated without a bench, and — the real
point — so defects can be *injected with known ground truth*, which is the only
way to know whether a measurement chain reports the right number.

It deliberately reproduces the artefacts specific to photographing a display:
perspective and lens distortion, backlight PWM banding whose phase advances per
frame, defocus, thermal drift, shot noise and 8-bit quantisation -- and
reflections off the cover glass (`Reflection`), which are added in linear light
and after the PWM banding, because the room's light shares neither the
display's encoding nor its backlight. (Added before the banding at first, a
reflection came out striped, and the glare correction was being tuned against
stripes no room produces.)
`BezelPanel`/`BezelRig` add a display set into a trim that can carry a ChArUco
board, which is what makes routes B and E testable at all.

**Numbers measured against it say the code is correct. They say nothing about
what a real rig can resolve.** That is what the repeatability study is for.

---

## 13. Testing strategy

Four layers, doing different jobs:

1. **Unit tests** — estimator behaviour, verdict ordering, tolerance arithmetic,
   profile round-tripping.
2. **Ground-truth tests** — inject a known defect into the simulator, assert the
   measured value matches. This is what makes the accuracy claims checkable.
3. **Architecture tests** — the ML line, the licence constraint (§3). These fail
   the build on a structural violation, not a behavioural one.
4. **Benchmarks** (`benchmarks/`) — not pass/fail. They produce the tables in
   [method.md](method.md) and each carries, in its docstring, the things that
   looked right and measured wrong.

The benchmarks are part of the documentation, deliberately. A design decision
recorded as prose is an opinion; the same decision with a reproducible table
under it is a finding.

---

## 14. Licence discipline

Runtime dependencies are Apache-2.0 / BSD / MIT only. No AGPL, no
non-commercial weights — this domain is full of both, and a validation tool that
cannot be shipped to a supplier is not a validation tool. SIFT is usable
(patent expired 2020, Apache-2.0 in OpenCV). `tests/test_architecture.py`
asserts that specific known traps have not appeared in `pyproject.toml`. See
[licences.md](licences.md).

---

## 15. Module map

```
src/layoutval/
├── types.py          core data model; every geometric quantity in display px
├── pipeline.py       the six stages, wired
├── capture.py        stage 1: frames, median stacking, settling detection
├── calibration.py    stages 2–3: intrinsics, six homography routes, drift
├── displayfind.py    proposes the display's corners from an ordinary photograph
├── glare.py          reflections off the glass, subtracted
├── blur.py           camera shake, matched between the two photos
├── measure.py        stage 4: estimators, per-kind measurement, guards
├── verdict.py        stage 5: ordered rules → verdict + reason
├── report.py         stage 6: JSON records and annotated overlays
├── residual.py       advisory: what you did not model
├── profile.py        the element inventory and its reference patches
├── teachin.py        differential teach-in against CAN signals
├── autoprofile.py    inventory from the reference frame, when none is authored
├── authoring.py      assisted annotation — the only place a model belongs
├── repeatability.py  σ, from a static screen
├── linearity.py      sub-pixel bias, which a static study cannot see
├── server.py         phone capture over LAN
├── simulator.py      test fixture with injectable ground truth
├── demo.py           the whole pipeline against the simulator
└── cli.py            entry points
```

Dependency direction: `cli` → `pipeline`/`server` → `measure`/`verdict`/
`calibration` → `types`. `authoring` is imported by nothing on that path, by
design and by test.

---

## 16. Where to extend

| you want to | do this |
|---|---|
| support a new element type | add to `ElementKind`, add a `measure_*` in `measure.py`, dispatch in `measure_element` |
| support a new cluster's calibration | add a `homography_from_*` returning `DisplayGeometry` with a new `method` string |
| import from a new design tool | add a reader producing `ElementSpec`s with `source="design"` |
| add a check | prefer a new ordered rule in `verdict.py` over special-casing a measurement |
| use a model for annotation | implement the `MaskProposer` protocol in `authoring.py`; do not import it anywhere else |

---

## 17. Known gaps

Stated plainly, because the alternative is someone discovering them later:

- **No design export is wired in for the reference HMI.** Until one is, results
  answer "matches the reference frame", not "matches the design".
- **Route E is validated against a fixture whose bezel is coplanar with the
  active area.** A real display is recessed behind glass, and a homography
  anchor is exact only for a coplanar pair. Expect worse on a real bench, and
  measure it there.
- **Route E is characterised for sampling ratio 0.5–1.3.** Above about 2 the
  panel stops fitting in frame.
- **`--lens-board` still requires a target that is not the cluster.** Deriving
  distortion from the straightness of the display's own border would remove it;
  the technique is standard (plumb-line calibration) and the edge points are
  already extracted, but it is not built.
- **Colour and symbol validation are thinner than geometry.** The HSV mask
  machinery exists for telltales but is not developed into a full colour check.
- **Glare is handled one photograph at a time.** A compact, strong reflection
  (a lamp, not a window) is only partly subtracted, and the identity re-check
  on detail covers for that, not the subtraction itself. The opening also treats
  as background a filled element wider than a tenth of the frame -- in both
  frames alike, so it cancels, but its interior is not measured. And an element
  a reflection whitened completely cannot be measured at all. The strongest
  remedy for all three is not built: several photographs from slightly
  different places, aligned in display space, and the darkest value each pixel
  takes across them -- reflections move with the camera and the content does
  not, and a reflection only ever adds light.

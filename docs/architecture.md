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

SAM 3 mode (§5, route F) follows the same line. The model runs on another
computer (`sam3_server/`); `sam3client.py` only sends it a photograph over HTTP
and fits a shape to the mask that comes back. What it produces is four
*proposed* corners that a person confirms, exactly like `displayfind.py`'s, and
a screen outline that narrows which part of display space is inventoried. It
never measures an element and is never consulted during validation, and
`tests/test_sam3.py` asserts the client imports no model.

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
edge sub-pixel. Where the points come from is a separate question with four
answers, and the phone can switch between them at any time (§11): a person taps
them, `displayfind.py` proposes them and a person confirms, or route A replaces
the lot when the cluster can draw its board. A fourth, with `--sam3-url`: SAM 3
segments the screen named by a text prompt and `sam3client.py` fits its outline
(see "SAM 3 mode" below).

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
- **With the resolution typed in, the screen's shape checks it.** The
  per-side answer is kept when its true proportions (below) are within 8% of
  the resolution's. Searching first and ranking by shape put one side on a
  laptop's lid in a real photograph: lid and screen together were nearer the
  typed-in shape than the screen alone was through the lens. Only when the
  per-side shape is wrong is the search below run. A cluster that
  draws large panels of its own defeats "innermost clean line": on a bench
  photograph of an EV cluster the proposal was a strip down the middle of the
  screen, and on a second shot nothing at all -- and with it an inventory of 5
  elements where there are 55. So every combination of the longest lines per
  side (14 each) is tried, and kept only if its true proportions -- recovered
  through the perspective by Zhang and He's whiteboard method, the principal
  point at the image centre -- are within 8% of the resolution's, every side
  covers its span, no side has more than 10% lit content beyond it, and the
  step across every side -- just inside against just beyond -- is alike to
  within 55 grey levels: one screen in one bezel steps alike all round, where a
  rectangle of the right shape patched together from a window's edges and a
  shelf on the wall behind stepped -10, -9, -11 and +118. Ranked by cleanest
  bezel, then shape, then evenest step, then coverage, then innermost. On six
  bench photographs it found all six screens, agreeing with the per-side answer
  where that was right; where nothing fits, the per-side answer is returned.

Measured on the two bench photographs under 120 random warps -- perspective,
rotation, exposure -- it found the screen every time, and was never confidently
wrong. It proposes on the undistorted frame and maps the dots back into the
photograph's own pixels: proposed on the raw photograph instead, lens distortion
bowed the edges and the corners landed 29-34 px off, far enough for route F's
snap to refuse them. Accepted unchanged, the proposal now snaps to within
0.03 px of route E's border fit. It is deterministic line geometry and sits on
the measurement path under the same no-learned-models test as everything else.

### SAM 3 mode: any shape of screen

The corner routes look for four straight edges; a round cluster has none. With
`--sam3-url`, the photograph and a prompt (`display screen` by default, or what
was typed on the page) go to `sam3_server/sam3_server.py` on a GPU computer,
and the mask comes back. `sam3client.fit_display` decides what it is:

- **Four-sided** when the quadrilateral fitted to it covers it to IoU 0.97:
  each side a Huber line through the middle 76% of that side of the outline
  (rounded corners do not pull it), corners where the lines meet. Then route F
  snaps them to the panel edge as it does tapped ones.
- **Round or oval** when its fitted ellipse covers it to 0.97: the framebuffer
  is taken as the square the round picture is inscribed in, mapped by the
  symmetric affine map from that circle to the ellipse. Not snapped -- there is
  no straight edge. A circle cannot say which way is up, nor, under
  perspective, exactly where its centre is (about r² tan(tilt) / distance off:
  10 px for a 600 px radius 3° off square), so the camera is assumed upright
  and square on; the dots can be dragged. The mapping is the same for the
  reference and every validation, so its error cancels in the comparison.
- **Anything else**: the smallest rotated rectangle round it, shown as needing
  checking.

At Reference, SAM 3 is asked again on the reference photograph, the mask most
over the calibrated display is taken through H into display space and, less a
thin rim, becomes the screen: for a round cluster the framebuffer's corners are
bezel and nothing there is inventoried or compared. Night or day theme is then
judged inside that outline only -- judged on the whole framebuffer, the black
corners round a circle read a night theme as a day one, and the glare step took
a plain block that moved 20 px for a reflection and passed it. If the GPU
computer is unreachable, the page says so and the corners can be tapped; the
reference falls back to the usual screen search.

### Drift

`DriftTracker` re-solves pose per frame against the reference. In
`MOTION_EUCLIDEAN` mode it corrects a nudge; in homography mode it absorbs a
full pose change, which is what makes a hand-held phone usable at all. The cost
is stated rather than hidden: a correction that re-solves the whole pose also
absorbs a fault in which *everything* moved together. Per-element faults survive
it because the rest of the frame dominates the fit; a whole-layout shift does
not. `--fixed-camera` checks pose without re-solving.

The hand-held re-solve fits the screen only (`CaptureSession._pose_masks`): the
display's quadrilateral in the reference, less the photo's undistortion border
and any reflection's footprint. A homography is exact for one plane. Fitted to
a whole photo with a backdrop at 1.4x the screen's distance, a camera moved 3 cm
sideways at 50 cm put every element of a good screen 6-11 px out; fitted to the
screen, the worst was 0.6 px. The empty border left by undistortion stays put
when the camera moves and was read as the scene staying put -- 0.8-1 px of pose
error for a camera turned 3 degrees or 10% closer. A reflection lies on the
glass, not in the picture -- unless it covers most of the screen, as a haze of
room light does: left out then, what remained was too little to fit, and a
good screen came back FAIL with the phone moved 4 degrees (5-6.5 px). It is now
left out only while 40% of the screen remains. If too little is left to fit,
the whole frame is used as before.

**The pose is finished on the elements** (`anchor.py`). The screen-wide pose is
fitted to brightness, and whatever changes brightness without moving the
screen pulls it. The elements lie on the display's plane and are each located
on their own detail, so after a first measurement the pose is corrected by the
one homography that best explains where they were found -- RANSAC, 1 px, so an
element that really moved is left out of the fit and still reports its move --
and everything is measured again. A homography needs 20 elements that shift
(icons, text, telltales, regions; not needles), an affine map 8; below that
the pose is left to ECC. With the elements to finish it, ECC runs coarse --
1000 px, 60 iterations -- which under haze was both better and faster than
2000 px (4.6 px off in 54 s with the camera not moved; 0.6 px in 2 s). If no
mapping explains most elements, ECC is run fine from the coarse pose and the
elements are tried again.

Measured on a simulated bench -- the user's cluster in a laptop window, a wall
behind at 1.5x the distance, a haze on the glass that moves with the phone:
turned 4 degrees, moved 4 cm sideways, or closer and tilted, a good screen
passes (worst 0.2-0.4 px) and a 2 px move of one text block reads 1.97-2.14 px,
on that block alone.

**A group that moved together is left out whole.** A homography can bend: at
one side of the screen, where only a gauge pins it, it can follow part of a
gauge that was drawn a few pixels off, and leave the rest of the gauge as the
only failures. Counting inliers prefers that, because the bent mapping explains
*more* elements than the right one. On the user's bench the right half of the
moved left dial failed and its left half passed; on two bench photographs with
the left dial moved 3-4 px, 66% of its elements were flagged and 86 elements
elsewhere were flagged instead, some with the camera not moved at all. So each
candidate mapping is scored by what it explains plus the largest set of the
remaining elements that share one displacement of at least 2 px: under the
right mapping the moved gauge is one such set, all of it; under a bent one its
displacements differ element to element. The winner is refitted with that set
taken as one rigid block with its own shift, so its shape helps pin the side
of the screen nothing else does. On a two-dial layout with the left dial moved,
the old fit flagged 0-21 of its 56 elements and 2-33 of the other 79; this one
flags all 56 and none of the 79, the move recovered to within 0.4 px.

**Every element votes.** Which set is the pose and which moved is decided by
count, so every element has to be counted. Measured as usual, an element is
looked for 8 px round where the first pose puts it, and the first pose can lock
onto the moved group: on a webcam shot with the right of the screen hazy, the
sharp left dial was what the matched features and ECC followed; the rest of the
screen sat 8 px or more off, came back "missing or displaced" and had no say,
and the moved dial passed while everything else failed. So before anything is
measured, every element that shifts is looked for up to 32 px away by a plain
ZNCC match (`anchor.far_correction`), and the pose is corrected from those
votes. On a synthetic screen with the rest 12 px off, an 8 px search let 28 of
60 elements vote and the moved group won; 32 px let all 60 vote. On two bench
photographs with the first pose forced onto a dial moved 6-12 px, 2 of 12 runs
had flipped; none do now -- all 434 of the dial's elements flagged, and 36
elsewhere where there had been 165. The votes are taken again under each
correction until it settles, each round starting from the last one's choice of
which elements are the pose and which moved; chosen afresh, a near tie could
swing a later round back to a bent mapping.

**Few elements, and more than one thing moved.** A sparse cluster -- 32
elements -- with its speed band moved 4 px and the speed digits and gear letter
3 px left 19 unmoved, one short of the 20 a homography was allowed from: no
correction was made, the pose stayed where ECC put it, pulled 1-2 px towards
what moved, and half the band passed. Left alone, the gear letter left 20 and
it worked -- which is how moving one element changed the verdict on others.
Now fewer than 20 get an affine map, which cannot bend; groups may be more than
one, moved by different amounts, each a rigid block in the fit; and the
mapping must hold more elements than any group, since the camera is what most
of the screen agrees on -- without that, a mapping lined up on what moved
scored as well as the right one. On hazy webcam-size frames of a bench
cluster cut to 32 elements, the first pose pulled towards three moved things,
unmoved elements flagged went from 35 of 171 to 4.

**The same scene twice gives the same answer.** Nothing in the measurement is
random -- the RANSAC is seeded, SIFT and ECC are deterministic -- so two
captures differ only by what the camera does between them. Simulated on a
bench photograph at webcam size, 8-10 captures each with fresh sensor noise
and compression, exposure wandering 12%, white balance 6%, focus hunting
between 0.3 and 1.6 px of blur and focus breathing 0.4%: on the full
inventory an unchanged screen never changed verdict, and a moved gauge
changed one borderline element. On inventories of 32 it was not so: on one
of six, one capture in ten lost its way in a later round of element votes --
a few elements dropped out of the mapping, it fitted the rest worse, more
dropped out, 21 down to 11 -- and the pose went 11 px off, seven unmoved
elements failing. So a later round may only refine (at most 1 px), keeps
the first round's kind of mapping, and each fit returns its best-scoring
state rather than its last. On those six inventories, the elements that
changed verdict between captures went from 8 to 1 -- one sitting on the
1.5 px line beside the moved gauge.

**Two answers that fit alike.** On a sparse cluster through a mounted webcam,
with a reference taken once and the same faulted screen captured again and
again, the first few came back right and later ones turned it inside out: the
moved speed band passing, the unmoved km/h, battery and clock beside it
failing. Nothing is learnt between validations -- the reference, the
templates and the calibration are only written by Reference -- so it was what
changed in the room: a reflection growing on the glass. Rebuilt in
simulation (a BMW-style layout at 0.51 camera px per display px, haze, moire,
a reflection growing over two minutes), one capture in nine flipped. Three
things were behind it:

- ECC, pulled by the sharp moved band, already bent the first pose 3.5 px
  towards it, so the band read 0-1 px and the rest carried a stretch.
- At that sampling each element is placed to 0.5-1 px, about the inlier
  tolerance, so the right answer lost supporters to noise, and a mapping drawn
  from four sharp elements of the band and the right side won. Now the best
  40 distinct candidates are refitted to what they agree with before they are
  compared (locally optimised RANSAC), and every element's own shift is a
  candidate too: a shift cannot bend.
- Even so, two explanations fit every element: the band (11) moved 4 px, or
  the view stretched 0.3% and the four elements beside the band moved 4 px the
  other way. Counting inliers picked the second. Now, of explanations within
  two elements of the best, the one that leaves the screen nearest where it
  was taught is taken (`home`): a mounted webcam's screen has not moved. On
  synthetic votes of that layout with the first pose stretched, wrong answers
  went from 28 of 30 to none at 0.2 px of vote noise, 10 to 1 at 0.5 px; at
  0.7 px, 17 to 13 -- the measurement there is too coarse to tell.

**A camera held still.** A webcam on a stand, a reference taken once, and the
same faulted screen captured again and again came back right, then with the
moved speed band passing and the unmoved elements beside it failing, then half
of each. The same photograph always gives the same report -- nothing is learnt
between validations, the reference is never written by one, the RANSAC is
seeded per call -- so it was what each capture's noise did to two decisions.
First, the pose was re-solved from what was drawn: with the band and the media
area moved 6-7 px (17 elements) against 23 that were not, noise tipped it
towards the band, or halfway, and the screen's own outline, drawn through that
pose, was 10-15 px off. So the photo is first lined up on a band round the
screen's outline alone (`_held_pose`: ECC from where the screen was, on the
bezel, the window frame and the wall round it, and on their detail only -- a
reflection nine grey levels high over the dark bezel had pulled it 1.5-1.8 px,
and 0.04 px with its slope of light taken off); when that converges
(correlation 0.8 or more) and no corner moved more than 4 px, it is the pose,
and the element votes may refine it by at most 1 px. A phone moved between
shots or a tilted lid moves the outline further, and a photo smeared by a
moving hand (its detail along the outline under 0.7 of the other's) lines up
1-1.5 px out; both go the usual way. Second, the camera-shake match estimated
its smear from the whole frame, and the moved group put a ghost in it 6 px
from the true defocus; centred on the centroid of both, it shifted the
reference 1.7 px -- only on the captures a touch softer than the reference.
Now, with the camera held still, every element is looked for first; three or
more that moved together, 4 px or more from the common shift, are a group, and
whatever of the estimated smear lies nearer the group's displacement than the
common one is cleared before it is used. (Leaving the moved elements out of
the frames instead destabilised the estimate, 5.9 px of smear for 2.3. With
the camera moved, the smear can be wide and an uneven screen's own 3-9 px
shifts overlap it, so nothing is cleared: cleared there, real smear went with
it and 11-20 more labels failed.) Simulated from the user's own photo at
webcam size, the camera still, fresh noise, exposure within 3% and focus
within 0.5 px each capture: 2 of 12 captures had flagged 17 unmoved elements;
none do now.

**A first pose that missed the screen.** Six phone photographs of a laptop's
cluster, nothing on it changed, from in front, well above, below and either
side. From well above and from the left, matched features offered no start
(15-21 agreed on the right homography; 25 were asked for), ECC began from
nowhere and settled on the keyboard, and every element failed. A blurred one
started 8-30 px out; each element then matched 1-2.5 px loose and no mapping
explained them to 1 px. So when the first round of votes finds no mapping,
several starts are tried: the first pose, the homographies fewer matched
features offer -- on 1000 and 2000 px copies, since a close-up against a view
from the side matched enough only on the larger -- the same on
contrast-equalised (CLAHE) copies -- which found the
screen seen from the left where the plain match put a corner 130 px off -- and
the screen's outline found by shape in both photographs. Each is brought closer
by the homography most elements agree with to 3 px (`anchor.rough_correction`),
voted on again until it settles, and the one most agree with to 1.5 px is kept;
the usual fit then decides what moved. A start that is wrong is outvoted, not
trusted. A first fit after that which moves the pose more than 3 px has lost
its way among loose votes and is not taken. Failing elements on those three
photographs: 32 of 32, 34 of 35 and 13 of 14 measured, to none of 35 --
REVIEW, since from that far round the run cannot be judged at 2.5 px (below).

**A screen not quite flat** (`viewpoint.py`). With the pose right -- from the
screen's corners, fitted to every element -- the middle of the laptop's screen
was still 3-3.5 px out against its sides, in four of five pairs along the way
the camera had moved:
3.5 px left with the camera moved left, 3 px down with it moved down. That is
parallax off a lid about a millimetre out of flat, and no homography removes
it. So each camera's position over the screen is recovered from its pose and a
focal length (the lens solve's, or 0.6 of the frame diagonal: a phone's main
camera, a laptop webcam), and at each element the shift a screen 0.4% of its
width out of flat would show is worked out, with its direction. A position
failure that is no more than that, along it, is REVIEW with the reason
`position_view_angle`; across it, or by more, it fails as before, and nothing
is passed this way. Labels seen from that far round were also compared
softened (1.5 px blur): one photograph showed thin strokes on a hazy screen,
the other the same strokes bold and edged by the phone's sharpening, scoring
0.69-0.77 for the same "50". Softened, 16 of 24 matched, while of 399 pairs of
different elements only one more reached 0.80 (from 0.78). When the
unevenness at the screen's centre is more than the failure tolerance, the run
is REVIEW and says how close to come: on that laptop, 13-14 degrees round
(2.0-2.1 px) a 4 px move of the speed band was found whole; 25 degrees round
(3.5-3.8 px) it was found as REVIEW at best, and once not at all. Below 1 px of
allowance -- a camera that stayed put, or moved a little -- none of this
applies.

The page also fixes the webcam's exposure, white balance and focus once it has
settled (2.5 s after it starts), where the browser and camera allow it --
Chrome does, on most USB webcams -- and says which it locked, or that it could
not. Then an exposure slider sets the exposure time by hand where the camera
allows it (else its exposure compensation), on a log scale shown in stops from
the camera's own choice, with whatever else is locked kept locked; it is
remembered per webcam and re-applied when the camera opens, so reference and
validate are taken alike.

`--edges` (`edgecheck.py`) measures what the re-solve cannot: the drawing
against the display's edges. The edges are proposed in the reference; at
Validate each is carried into the test photo through the solved pose and a
strip across it is lined up with the same strip in the reference (ZNCC over
1-D profiles, groups of 8 along the side, median). An edge that moved against
the pose means the drawing moved against the edge. Lining up needs no clean
step -- no side of two bench photos had one -- and the strip reaches only
0.15 of its width inside the edge, or the drawing there pulls the answer to
zero (0.35 px read for a 2 px shift). Opposite sides must agree within 0.5 px;
the shift is the elements' median movement against the pose less the edges'.

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

What is kept is set by what a hand-held photograph can measure. A region must
be at least 40 px and 9 px across its narrower side -- a speedometer's tick
marks (48-66 px) came in when the floor dropped from 80 px, and so did marks a
few pixels thick that a slightly shaken photo smears past recognition (19 of
127 elements of 30-80 px failed at least once under a 12 px shake streak;
thickness decides it). It must carry crisp detail somewhere -- the 98th
percentile of stroke-scale contrast, at least 0.4 of the frame's median, so a
dial's sharp ticks count against the soft glow round them while a faint patch of
a bar's gradient fill does not. And it must keep 2% clear of the frame's edge,
where the display's rim and a laptop's status icons live and a hand-held pose
is least certain. Up to 200 elements are kept.

A webcam on a stand does not shake, and the one-button run says so
(`steady_camera`): there a region need only be 20 px and 4 px across, which
is what a dial's minor ticks are. Through the page, on the simulated cluster,
the layout check went from 54 elements to 112, every one passing. And a short
crisp stroke is kept although its detail is mostly one-dimensional
(`short_strokes`): a tick's two ends fix it along its length, but against its
long sides they score low, 0.035-0.08 on a real webcam's photograph, where
nine of the twelve regions the 0.10 cut took out were ticks. Kept are strokes
scoring at least 0.03 (lines and edges score under 0.02), no longer corner to
corner than twice the screen's typical element (a needle, a bar, an arc are
longer), and as crisp as 0.8 of the typical element.

**Big, mostly empty regions are split.** One threshold for the whole screen
joins whatever touches at that level: a dial's scale ring joins every tick and
numeral along it, and an overexposed camera's bloom joins neighbouring white
strokes through the glow between them. On a real webcam photograph of a
cluster, blown out, the inventory had boxes a quarter of a dial wide and no
20, 40, 60, 80 or 140 at all. What is big is judged against the typical
element on the same screen -- the median box, once there are 8 or more --
not against the frame, so black padding round a display changes nothing and a
screen with three words on it has nothing to split. A region is split when its
box is 25 times the typical element's and less than 30% of it is lit (a ring,
not a block), or 60 times however full it is (a dial face lit by the bloom,
numerals on it). The split tries, in turn, thresholds 30%, 45%, 60% and 75%
of the way from the box's background to its brightest strokes -- bloom is
dimmer than the strokes it joins, and the levels are the same whatever is
round the box -- and, for a sparse region, opening away lines thinner than a
stroke (a scale's ring is a hairline; its ticks are not). The first try giving
three or more element-sized parts that hold at least 30% of the region's light
is taken, and a part still big is split again, three levels down. On the
blown-out photograph, with the overlay cleaned off, 72 elements became 135, and
the left dial's numerals came out one each. Splitting needs nothing from the
camera, so it also covers a webcam that offers no controls.

### Teach-in over CAN: lamps by symbol, elements by sweep

`canbus.py` is the tester's side of the bus: every signal of a DBC held and
every message sent cyclically, as a vehicle network does, with a changed
message sent at once. `lamps.py` runs a **teach plan** through it (the HMI's
`can/teach_plan.json`): rest, each lamp alone, all lamps together, each
readout swept through a few values. The server runs the plan from the page,
because the webcam is the browser's: per step, set the bus, wait, upload the
shots. Every shot is rectified with the calibration and saved, so a teach-in
is a record as well as a result.

What lit up is found as a **lit change** against the rest shot, not a
difference: brightening, or taking on a lens colour (chroma weighted 1.4,
because a dim red lamp on grey moves the colour far more than the value). A
threshold below the rest screen's own variation -- the top of its difference
with itself over five shots -- would find the camera's noise and the
backlight's banding. Strokes a quarter of a symbol apart are one blob.

**Lamps are kept by symbol.** The HMI's strip shows only lit lamps, centred,
so all 31 lamps lit alone light in the same middle slot: where a lamp is says
nothing about which it is. Each lamp's template is its lit change, from the
shot it is brightest in -- judged in its own box: over the whole frame a
camera's noise outweighed one small lamp, and two flashing lamps were learnt
from a shot they were dark in. Reading a frame is template matching on its lit
change, each lamp placed at most once, best first, and only where:

- about as much light falls as the lamp gives (half to double): the
  normalised score saturates over a nearly flat patch, and an empty corner of
  the change map "matched" a lamp at 1.0;
- no other lamp's symbol fits clearly better, even one placed elsewhere: a
  flashing seat-belt lamp in its dark half otherwise settled on the airbag
  lamp beside it;
- no lamp's symbol fits clearly better in shape with only its colour wrong:
  ABS drawn red is ABS in the wrong colour, not the red brake lamp its circle
  resembles.

The light bound is 0.3, not 0.5: a capture is a short mean of frames, and a
flash that switches during one leaves the lamp half lit. Run end to end
through the page, the browser's webcam and the HMI over CAN, every flashing
lamp came back half lit in some shot, and at 0.5 each was "unknown" -- 7 of 33
lamp states failed on a correct screen. And a flashing lamp dark in a shot can
be read, at a clearly lower score, onto another lamp its symbol half fits --
front fog, which the HMI draws over the gear letter -- so between shots, a
lamp's readings away from its best one and clearly weaker are dropped as
misreads; a reading elsewhere that matches as well means the lamp moved.

Colour rules a match out when two hues differ by more than 18 (of OpenCV's
180), and costs it 0.3 when one is white and the other coloured, so two lamps
sharing a symbol in two colours (cruise set and ready) are told apart. Below
0.82 a lit symbol is **unknown**, with the lamp it is nearest. Shots are read
one at a time and then put together: a lamp lit in some and dark in others is
flashing, and one whose place differs between them moved -- reading a stack of
shots as one picture had shown two layouts on top of each other when the HMI's
strip reflowed with every flash.

**Elements are named by sweep.** An element of the rest screen belongs to a
sweep when its own box changes in most of the sweep's states (12% of it at
least): digits change in every state, while a needle crossing a scale numeral
changes it in one at most. Names replace the auto inventory's ids
(`Speed`, `Speed 2`, `unknown@312,75`) and are applied again whenever the
reference is retaken.

**Checking** compares what is lit with what was sent. Places are checked only
in the states lamps were taught in -- alone, and all together, against each
lamp's slot there -- because elsewhere the strip lays them out differently; in
any state, the order along the strip is checked against the all-lit order,
and only lamps outside the longest run in order are flagged, so two lamps
swapped flag two, not the strip.

### Discovery: the teach-in from the DBC alone

`discover.py` replaces the teach plan with the matrix. `canbus.Rig` works the
bus out of the DBC (cluster node, messages and cycle times, ignition, E2E
counters and CRCs, quiet rest values; `rig.json` corrects any of it), and
`Discovery` lists every signal the cluster receives with the states to try:
each bit flipped, each value of a value table but the ones meaning "not
available", each number at 5, 25, 60 and 95% of its range -- ordered away
from the end its rest value is nearer, so that it comes back to rest from the
far side. A warning with hysteresis is why: the HMI's low fuel lamp lights at
12% and lets go at 15%, and its DBC's rest value is 14%, so a sweep that
ended low left the lamp lit for the rest of the run.

Each signal is looked at as soon as its states and the rest after it are
shot (`look`), on a thread of its own while the next signal is shot, and its
frames are let go; `learn` then puts the signals together. A **lamp** is a
symbol lighting where the rest screen was empty -- 90% of it on pixels with
nothing drawn within 3 px -- because a letter turning into another (P to D)
lights pixels the old one did not use. A number's symbols are lamps only in
one of the bulb check's places (`bulb_slots`: compact symbols lit throughout
the bulb check where the rest screen is dark), since a readout's digits come
and go in dark places too. One lamp is one symbol in one place and colour,
whatever lights it: ABS On and Flash, every door, the coolant request and
the coolant temperature at 95%. Two learnt templates are both noisy and
match each other less well than either matches a frame (0.74-0.78 for the
same lamp through the page), so one lamp is a symbol within a third of a lamp
of the other, in its colour, matching at 0.6. A lamp is flashing when it went
on and off at least twice across a state's shots -- a flash does, over more
than one period -- not merely when it was dark in one: dark in the first shot
only, it came on late. Each signal's change, less its lamps, names
the rest screen's elements (`name_elements`), and what changed between the
rest before a signal and the rest after it changed on its own (`_drift`):
the elements there -- the odometer, the trip -- are left out of the layout
check.

### The one-button run

`autorun.py` runs the whole sequence on a thread of its own, fed by the
page: the page only streams the webcam's frames (`/auto/frame`) and shows
what comes back. Nothing is told which frame to use.

**`watch`** decides that, for every state sent: frames arriving within the
run's own delay of sending are of the state before -- the delay from a state
sent to the first frame showing it, measured at ignition off and on (the
longer of the two, between 0.12 and 1 s; through the page and the simulated
webcam it came out at 0.14-0.44 s, which is why it is measured, not fixed);
a change is a frame unlike every
frame the screen showed before (one, or both halves of a flash); settled is a
frame matching one 0.25-1.2 s older -- still, or a flash come round. Then
shots are taken across 1.8 s, or 0.7 s when nothing moved in that time (no
flash has longer halves) and at least five frames came: through the page, a
real webcam's frames came 0.6-0.7 s apart, two of them both landed in the lit
half of a 0.76 s flash, and the high beam's Flash state was taught as steady.
Five frames with the page's varying gap between them do not stay in one
half. A state that changes nothing is given 1.5 s. Frames
are compared small (480 px wide), and a pixel has changed when it moved by
more than the noise and by more than 30% of its brightness: a backlight's PWM
banding through a camera is a gain on each row, ±8% in the simulated camera,
which is 40 levels on white text -- with an absolute threshold the screen
never stood still. The noise is measured once the picture is still, since the
cluster may be part way through starting when the camera opens.

Frames at a steady rate near a flash's half period all land in the same half,
and a flashing lamp then looks steady, or is never seen lit: through the page
the frames came about every 0.35 s and the HMI flashes in 0.38 s halves, and
the traction-control lamp's Flash state was "missing". So the page leaves a
varying gap between frames (60-180 ms on top of the time a frame takes), and a
lamp-check state that does not pass is looked at again for 2.5 s before it is
reported ("seen twice"). A symbol that matches nothing, in the place of a lamp
the state lights, is that lamp at another brightness: the HMI's shift prompt
fades between 15% and full rather than flashing, and at a quarter of its light
it no longer matches its own symbol. A wrong symbol in that place still leaves
the lamp missing, so the fault is not lost.

**The power cycle** gives the display and the bulb check. Ignition off until
dark and still (and the bus asleep, if that is what it takes); ignition on,
recording ten frames a second, until the screen has stayed still for 6 s --
a bulb check can stand still for several seconds, and was taken for the
running screen when 2.5 s was enough. What lit up locates the display
(`lit_region`, over the running screen and the whole start-up: the bulb check
lights lamps the running screen leaves dark, and on the BMW design's black
screen the outline round the gauges alone cut the top row of lamps off; the
corners finder's outline must hold 97% of it, else the outline is taken round
it), and the start-up frames with the most compact
symbols lit where the running screen is dark are the bulb check
(`bulb_frames`). A bulb-check symbol is an unknown lamp only if it is lit in
every bulb-check frame, still, sized like the lamps, and on no lamp: the
needles sweeping through the bulb check made twenty "lamps" on the dials
before.

What lit up is what got *brighter*, not what changed: with the screen dark a
webcam still on automatic opens up and the whole room comes out brighter, and
taken either way that room was "the display" -- on a real webcam the box
round it took in a lamp on the wall and came out turned, and the picture
through it was rotated and skewed though the camera was level. Where the
corners finder finds no outline (a black bezel in a dark room, the usual case
for a real cluster), the box round the content is upright unless a turned one
is at least 15% tighter (a camera rolled well over); its corners are not
pulled onto edges, since the edges next to them are ticks and numerals; and
it is taken at one display pixel to a camera pixel, not stretched to a common
resolution -- the content's shape is not the display's, and 16:9 made
1920 x 1080 of a 2.6:1 cluster's content. The size is estimated again on the
second look, with the camera set up. Content within 1% of the picture's edge
(`cut_edges`; of what there is of the picture, when undistorted) means part
of the display is out of view, and a teach-in stops there and says which
edge, after 20 s rather than after a quarter of an hour.

In the bulb check every lamp is lit at once, and each is read by its template
-- the symbol and a margin -- with the light in it held near the lamp's own.
From farther off the margin is wider than the gap between lamps, the
neighbours' light falls in it, and the lamp is passed over: a real webcam's
bulb check lost 11 of 31. A lamp not read among the rest is looked for where
it lit when taught, by its symbol alone (`Library.at_place`: within 30% of
its size, matched to 0.7, in its own colour, 0.3-3 times its light); lit
glow strong enough to lose 4 of 6 lamps to the blob reading lost none to it,
and a lamp drawn missing stayed missing.

**Setting the camera** (`camtune.py`). A webcam on automatic exposes for the
room, and in a room a lit display is the brightest thing: blown out, its white
strokes bloom into each other, its white balance follows whatever else is in
view, and its sharpening rings round every stroke. The page reports what its
camera can be set to (W3C Image Capture: `getCapabilities`/`getSettings`) when
the run starts; once the display is located, `Tuner` sets the camera for the
display alone, measured on what lit up at ignition on (in display space, with
a margin for the edges a focus measure needs). Not on the whole display
region: a lamp's reflection on the glass, or a lamp in the room that falls
inside the outline, does not change with the ignition, and one that clips at
every exposure a display wants would have the exposure cut until it did not
-- in the simulated webcam, a reflection that leaves the exposure at 95.7
measured this way cuts it to under half measured over the region:

- *sharpening* to its minimum, white balance and focus to manual at their
  current values, so nothing moves under the measurement; and *gain* (`iso`)
  to its minimum where the camera offers it, as a camera lowers its gain
  before its exposure in a bright scene -- a webcam left on automatic in a dim
  room has it up, and the exposure the display then needs is shorter than it
  has to be (8 times, at ISO 800 against 100), noisier and more open to
  flicker. Most desktop webcams do not offer it to a page;
- *exposure* (`exposureTime` with `exposureMode: manual`, else
  `exposureCompensation`): the brightest content is the 99.8th percentile of
  the brightest channel -- a display is mostly dark, its strokes a few percent
  of it, and the brightest channel keeps a red lamp's red from clipping -- and
  is brought into 200-240 with at most 0.05% of the display at 250 or over.
  Clipped, the level says nothing of how far over it is, so the exposure is
  halved (or brought halfway back to the setting last known too dark); below
  clipping a camera's response is near enough straight, and one step of
  225/level lands on it. Exposure is held under one frame at 30 fps
  (`exposureTime` 333), past which a webcam drops its frame rate;
- *focus*: seven points across `focusDistance`, then four finer ones round the
  best, by the variance of the Laplacian;
- *white balance*: red over blue of what on the display is bright, not
  saturated and not clipped -- its white and grey text -- at two colour
  temperatures 800 K apart, and the temperature where it is 1 solved for;
- *exposure* again, four steps at most, since white balance moved the
  channels.

The run says which settings it wants by number (`camera: {seq, set}` in the
progress the page reads back); the page applies them with `applyConstraints`
(one at a time if the camera refuses the lot, so one it does not take does not
keep the others off) and sends every later frame with the number of the
settings it was taken with (`X-Camera-Seq`) and what the camera says it took
(`X-Camera-Settings`). A frame is used only when it carries the latest number,
and only 0.6 s after the first one does -- a webcam takes a few frames to move
its exposure and longer to move its lens. A page that has not applied them in
8 s will not (an older page, or a browser without the controls), and the run
goes on with the camera as it is; so does one whose camera offers nothing.
What the camera reports it took, where it reports it, is kept rather than what
was asked (a camera rounds to its steps), written to `camera.json` with each
step, and put on the camera before every later run of the same session. After
tuning, the start-up is recorded again and the display located again with the
camera as it now is: focus moves the picture a little, and the reference must
have the exposure the test will.

Through the page, with a simulated webcam that starts blown (exposure 260, a
colour cast, focus off, sharpening on), it came to exposure 86.5 (brightest
content 221, nothing clipped), focus 0.417 against a true 0.42, white balance
5824 K (red over blue 1.004) and sharpening 0, in about 28 s. Saturation and
contrast are left as the camera has them: neither moves a measured position,
and a symbol's colour is judged against a reference taken with the same
camera.

**Teach, then test.** The first run teaches: display, camera, reference,
discovery, lamp check, validate. Later runs test against it: camera set as
taught, power cycle (every lamp lit in the bulb check, in its place), lamp
check, validate. The lamp check starts
with the rest screen as it is now and measures against that, not the
teach-in's -- minutes or days old; and after a lamp a number lit, the number
is driven past where it lit (a "letting go" step, not checked), for the same
hysteresis.

**Every element on its own** (`elementsheet.py`). The overlay labels every
element in place, and at a cluster's density the labels sit on each other. So
the run ends by cutting each element out by itself -- from the reference, and
from the frame validated, at the place it was measured at, its box drawn in
its verdict's colour -- and each lamp from the lamp-check shot it is
brightest in, and writes them against their names: a PNG each in
`elements/`, and `elements.html`, one page with the pictures inside it,
failures first. The page is served with a policy that lets it run nothing
and load nothing but its own pictures. Making it can fail without failing the
run: it is for people, never the verdict.

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

**Only where there is something to compare, and something to find.** With the
phone moved between shots, a bench run came back REVIEW with all 109 elements
passing: the display corners took in a strip of the laptop's lid and the wall
behind, which slide against the screen when the camera moves. Three rules, all
about what is not the display:

- the comparison covers only what both photos saw of the display, less its
  outer 2% -- the rim the inventory also leaves out -- and, where the corners
  were set past the screen, only the screen: at Reference the screen is looked
  for inside the corners by its shape, and used when it trims a margin (70-99.5%
  of the corners' rectangle), for the comparison and for the inventory alike;
- a finding that is the reference's own content found nearby (ZNCC 0.8 within
  4% of the display) is something moved or re-lit, not drawn or gone: the
  element checks measure moves, and anything else that moves is off the
  display's plane;
- a finding mostly outside the cluster's drawing -- its elements' span, padded
  3% -- where the reference itself showed detail is trim, lid or background.
  Outside the drawing on plain background it is kept: something drawn where
  nothing was is exactly what this check is for.

Set-aside findings go to a note in the saved report.

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
(tapped corners), `chessboard` and, when started with `--sam3-url`, `sam3`
(corners from the screen SAM 3 finds; the page then shows an optional prompt
box, sent as the `prompt` field with `propose`), live. The corner modes map into the screen's
full resolution and the chessboard into the board's canvas, so a switch is a
different display space: it starts calibration over, drops the reference and a
discovered inventory, and keeps the lens solve, which belongs to the camera.
Choosing the mode already on is not a switch. (Left calibrated across a switch,
the page saw no reference, moved straight on to Reference, and the chessboard
photograph meant to calibrate became the reference.) The chessboard mode needs
the exact corners the cluster drew. With `--board` they come from the HMI's
export; without it, `layoutval go` applies the HMI's own layout rule
(`hmi_board`: 100 px squares, one square of quiet zone, centred, origin on a
whole pixel) to the resolution typed in on the phone -- 14x5 inner corners on
1790x870, 16x4 on 1920x720. Checked against the HMI's own `pattern_geometry`,
corner for corner, and measured at 0.02-0.05 px against the simulator's true
mapping.

**The resolution is typed in on the phone,** at the top of the page, and
nothing but the lens solve runs until it is. It is not read from anywhere: the
computer running layoutval is not the cluster, so its own screen size (which
`go` used to read) is right on a laptop test bench and wrong on a car. Every
result is in the cluster's pixels, and a size of the wrong shape stretches one
axis -- 1920x1200 applied to a 1920x720 cluster reads a 2 px move as 3.3 px.
A board file for another size is refused, and a new size starts the
calibration again, keeping the lens.
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
- **Re-check identity on detail.** A compact lamp is
  only partly subtracted -- the opening cannot follow the top of a hill much
  narrower than its square -- and leaves a slope of light, which correlation
  reads as different content; haze too faint to be found as glare, and the
  phone's local tone mapping, do the same anywhere. An element that fails its
  identity check is compared again at the measured position on its fine detail
  alone (each patch less a blur of itself) and without the pixels a
  reflection clipped. Over a good screen that turned a FAIL into a PASS; a 2 px
  fault under the same lamp still measured 2.03 px, and a wrong symbol under it
  still failed. It does not let a different element through: of 406 pairs of
  different elements from two real clusters, none that failed on the whole
  patch passed on detail. Only under a reflection may
  an element too faded to tell be judged on position alone; elsewhere a faded
  or missing element is a failure.
- **Never a verdict of its own.** Glare is the room, not the display. It is
  subtracted and the elements are measured; it does not put REVIEW on anything.
  (An earlier version did, for anything a reflection clipped and for every
  result against a reference with a strong reflection on it; on a glared bench
  photograph compared with itself that was 75 REVIEWs, and the rule was wrong
  for what the rig is for.) Residual-check findings that sit on a subtracted
  reflection become a note: the reflection's photon noise stays behind after its
  light is removed. How much light was taken off is a note in the saved report.
- **What a phone does on top.** A phone does not record the added light and
  stop: under a bright glow its local tone mapping flattens contrast, and its
  noise reduction smears what is faint. Two bench photographs put through a
  glow and that processing came back FAIL on 19 elements that were still
  visible, "wrong content" at 0.19-0.78, positions exact. At that point no
  comparison can decide identity -- the same symbol, faded, and a different
  one scored alike -- so an element under a reflection whose contrast has
  fallen below 60% of the reference's is judged on position, which is still
  measured to a fraction of a pixel, and one line says what it shows could
  not be confirmed. A wrong symbol that is still visible keeps its contrast
  (161-172%, lamp or not) and stays a failure. The same photographs also came
  back REVIEW for reflected room objects -- a shelf, a lamp -- smaller than the
  opening's square and so not subtracted. They are out of focus, where
  anything the display draws is sharp: steepest slope over size of change,
  0.07-0.08 for the reflections, 0.26-0.33 for a stray mark drawn on the
  display, cut at 0.15. With both, the 16 photographs went from 8 FAIL and 3
  REVIEW to 16 PASS, and every injected fault was still found.
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

**Framing.** The display's right side kept coming back black, more on each
attempt and in every calibration mode. Undistortion was cropping each photo to
its all-valid region (`alpha=0`), and with a hand-held lens solve whose centre
is a little off that crop is lopsided: 6% of each edge strip for a good solve,
up to a quarter of one side for a loose one, gone before any other step ran.
The capture server now keeps the whole photograph (`alpha=1`), and the setting
is stored with the solve (`Intrinsics.alpha`) so a saved calibration stays true
to the frame it was solved in. What a photograph genuinely leaves out is said
rather than failed: the reference reports how much of the display is out of
frame and on which side, and at Validate an element outside the test shot is
not measured -- it would otherwise come back FAIL, missing, which it is not --
but named, with a REVIEW, so a partial photo cannot pass as a whole one.

The hand-held pose re-solve also starts from matched SIFT features now
(`feature_homography`, RANSAC over Lowe-ratio matches) before ECC refines it.
ECC walks downhill in brightness from wherever it starts, and a phone aimed a
sixth of the frame away between shots sent it to a wrong alignment that it
reported as converged -- every element FAIL. Seeded, shifts of 60, 160 and
260 px align. A mounted camera is not seeded: one that moved that far should be
caught, not followed. The reference's points are taken from the screen only
(the reference's screen quad), and the whole frame only when the screen has too
few: with a laptop's lid tilted between the shots and the keyboard and room
where they were, the keyboard's points matched as well as anything and agreed
that nothing had moved -- the hinge's corners exact, the top ones 400 px out,
and the pose solved from there, nearly every element failing. From the screen
alone the seed was within 0.2-4 px, and 109-111 of 111 elements passed with the
top edge lifted by 12-40% of the screen's height and widened by 4-16% each side.

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
- **The lens needs the board in every part of the frame, and 15 views.** A
  phone photograph of the lens board, fitted as a flat plane, left 1.1 px
  typical and 2 px worst in the middle of the frame; with radial distortion,
  0.13 and 0.25 -- a phone's lens after its own correction still bends, in the
  wavy "mustache" profile. That distortion is what made two bench photographs
  of one unchanged cluster, taken from different places, disagree by up to
  4.7 px at the dials' outer edges: corrected, 14-23 false flags per pair went
  to none, and 86.2% to 98.8% of elements right on every moved-camera pair.
  Simulated with that distortion, the lens solve depends on where the board
  goes far more than on how many shots: kept near the centre, 30 views still
  left 5-17 px at the edges; spread over the frame, 12 views left 0.12-0.83 px,
  15 left 0.08-0.36 and 30 left 0.05-0.30. So the phone asks for 15, does not
  solve until the board has appeared in each ninth of the frame (up to 30
  views), and says which parts are still missing.
- **The lens solve is gated on generalisation, not on fit.** Reprojection error
  measures how well the model fits the views it was given; a set of
  near-identical views is fitted beautifully and is wrong everywhere else, and
  reports a *lower* rms for doing so. Measured, a solve at 0.051 px rms left
  1.276 px of real error behind, and one that would have passed a 0.3 px gate
  was worse than skipping undistortion altogether. `LensCheck` therefore scores
  held-out reprojection and the spread of board angles, and a solve that fails
  is not adopted.
- **A repeated lens view is not counted** (`SAME_VIEW`): one whose board
  corners sit on average within 0.4% of the frame's diagonal of an earlier
  view's. Kept that tight because a tilt, which the solve most needs, moves the
  corners little -- distinct test poses 2-13 degrees apart were 0.54% apart at
  the least.
- **A lens solve is only applied to frames of its own size.** It is in one
  camera's pixels; a webcam frame after a phone's solve, or a phone turned on its
  side, is refused with a message rather than remapped into nonsense.
- **This computer's webcam** (`go --webcam`). The same page, opened at
  `localhost` -- a browser lends a page the camera only on a secure origin, which
  localhost is and a LAN address is not, so a phone never sees the option. A
  capture is the mean of 8 consecutive frames (`requestVideoFrameCallback`),
  sent as PNG; lens views are single frames, since the board is hand-held, and
  are taken hands-free: a 64x36 thumbnail of the live picture decides when the
  board has moved since the last view (mean change over 6 levels) and is then
  still (under 2.5 levels for 0.6 s). Its solve and captures go to `out/webcam`.
  Tested in Chromium with a fake camera playing a bench photograph: corners on
  the screen's edge, 108 elements, PASS; one lens view taken and no repeats
  while the board stood still.
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
├── sam3client.py     SAM 3 mode: the screen from a text prompt, any shape (HTTP only)
├── edgecheck.py      --edges: the whole layout against the display's own edges
├── anchor.py         the hand-held pose, finished on the elements themselves
├── viewpoint.py      how far round the camera went, and what that alone can shift
├── glare.py          reflections off the glass, subtracted
├── blur.py           camera shake, matched between the two photos
├── measure.py        stage 4: estimators, per-kind measurement, guards
├── verdict.py        stage 5: ordered rules → verdict + reason
├── report.py         stage 6: JSON records and annotated overlays
├── residual.py       advisory: what you did not model
├── profile.py        the element inventory and its reference patches
├── teachin.py        differential teach-in against CAN signals
├── canbus.py         the tester's side of a CAN bus, worked out from a DBC (Rig), sent cyclically
├── lamps.py          lamps by symbol, elements by sweep, and the lamp check
├── discover.py       the teach-in from the DBC alone: every signal tried, lamps and elements found
├── autorun.py        the one-button run: power cycle, display, camera, reference, teach, check, validate
├── camtune.py        the webcam set up for the display: exposure, white balance, focus, sharpening
├── elementsheet.py   every element and lamp cut out against its name: elements.html
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
- **The one-button run is measured only through the simulated webcam.** A
  real one has its own noise, auto-everything until the page locks it, and
  rolling-shutter flicker; the run measures its noise and its delay, but none
  of it has met a real camera yet. A cluster's signals that show only with
  another set first (the gear engaged shows in D) are reported as changing
  nothing until `rig.json` says so (`when`), and a lamp lit only by the
  cluster's own logic from several signals at once is found in the bulb check
  but stays an unknown lamp. A display's size worked out from its shape is a
  few percent off (the HMI's 1790 x 870 came out 1424 x 670 from the camera's
  side view), which scales every number but leaves comparisons with the
  reference -- taken through the same size -- as they were.
- **`--edges` is validated on photographs moved in software, not re-taken.**
  Two bench photographs warped to new poses, with the drawing shifted inside
  its edges, keep the edge in the plane of the pixels by construction. A real
  camera move sees trim or cover glass standing proud of the pixels shift
  against them by roughly that depth times the change in viewing angle -- a few
  tenths of a pixel for millimetres and a few degrees, more for a deep hood.
  That is why it is opt-in.

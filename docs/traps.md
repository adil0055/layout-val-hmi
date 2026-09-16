# Traps this domain is full of

Each one below is handled somewhere in the implementation; the pointer says
where, and the test says what it is supposed to do.

## Animations and transitions

Needles sweep, popups slide in, telltales fade. Never measure during a
transition. `wait_for_settle` gates capture on frame-to-frame difference staying
below a threshold for N consecutive frames, with a timeout that **fails the case**
rather than measuring whatever was on screen when time ran out.

`SettlingTimeout` propagates to a run-level flag with `severity="fail"`, and
`Pipeline.run` returns without manufacturing measurements for that frame.

Set the threshold from the rig: `estimate_noise_floor` on a static screen gives
the lower bound, and the smallest real animation gives the upper one.

## Day/night themes and backlight dimming

The same element at 20 % backlight has a different edge profile. ZNCC absorbs a
global gain and offset change; it does **not** handle a gamma change well. Teach
each theme separately — `LayoutProfile` carries a `theme` for exactly this, and
the simulator's night theme applies a gamma change as well as a gain change so
the failure mode is reproducible.

## Sub-pixel text rendering

Text drawn with sub-pixel anti-aliasing has colour fringes whose weight depends
on where the glyph falls relative to the display's RGB stripes. That shifts the
apparent centroid per channel. `ElementKind.TEXT` converts to luma before
measuring; never measure text position on a single colour channel.

## Gauge needles are not a translation problem

A needle rotates; it is not a shift-invariant patch and phase correlation on one
returns nonsense. `measure_needle` validates the pivot and the angle as two
separate quantities with their own tolerances and their own verdict reasons,
because a needle at the right angle about the wrong pivot is a real and separate
defect.

Two sub-traps:

- **The hub biases the pivot.** The tail extreme of the mask sits a hub radius
  past the true pivot — several pixels of systematic error that averaging never
  removes. `needle_pose` erodes the shaft away and takes the hub's centroid, and
  reports `hub_found=False` when it had to fall back, so the consumer knows the
  number carries that bias.
- **A luma threshold folds the plate into the needle.** A desaturated gauge plate
  inside the same box has a similar luminance to a red needle. Give the element a
  hue-and-saturation mask — `ElementSpec.mask` takes an `hsv_range`, reusing the
  masks already built for telltale colour validation.

## Elements that are supposed to move

Progress bars, scrolling lists, a fuel bar tracking a signal. Their expected
position is a function of state, not a constant. `PositionModel(kind="linear")`
models it; `TeachInSession.teach_moving` learns it from a sweep and records the
fit residual, because a large residual means the travel is not linear and the
model is the wrong shape.

Two sub-traps:

- **Diff the extremes, signed.** An unsigned `absdiff` between two sweep points
  merges the old and new positions into one blob whenever they overlap, and
  returns a box spanning the whole travel. Pixels that got *brighter* are where it
  travelled to; pixels that got *darker* are where it came from.
- **A mover needs its own stored template.** The reference holds the element at
  one state only, so at any other state the expected box is cropped from
  background and the element is reported as wrong content.
  `LayoutProfile.validate()` refuses a mover without a template, and
  `LayoutProfile.reference_values` records the state the reference was taken at.

## Z-order

Two elements each in the correct position, with the wrong one drawn on top,
passes every per-element position check that can be written. Two catches:

- the residual map, automatically but advisorily;
- `ElementSpec.occludes`, an explicit assertion that produces a real verdict,
  because the author asserted the relationship on purpose.

The assertion only means something between elements whose content is fixed — on
an element whose content changes with a signal it fires on the content change,
not on the z-order. It also refuses to judge a flat overlap: two uniform patches
correlate perfectly whatever their colours, so `check_occlusions` returns
`occlusion_indeterminate` rather than a confident wrong answer.

## Template anchors

A design export gives a *layout* box, padding included. A template taught by
toggling a signal is cut to the *ink*, which normally sits somewhere inside that
box. Differencing a position measured against one anchor with an expectation
expressed in the other reports the padding as a defect — on every run, for every
element, with a magnitude that looks entirely plausible.

`ElementSpec.template_anchor` records which, and `template_origin(value)` keeps
the two consistent as the element travels. During development this produced a
tidy, confident, completely wrong `+7.00 px` on one telltale.

## Reference drift

If the golden reference is a photograph and someone swaps the camera or moves the
rig, every reference is quietly invalid. If the reference is a design export in
display coordinates, it survives a rig change untouched. This is the strongest
practical argument for a design export over a teach-in capture, and it is why
`ElementSpec.source` is carried into every report — it is what decides what a
green result means.

The honest limit: without a rendering of the design you can only compare a live
capture to a reference capture. A design export gives *boxes*, so the absolute
check ("is the element's ink where the design box says") is limited by
segmentation accuracy, around half a pixel to a pixel. The per-run check is
reference-relative and an order of magnitude more precise. Both are worth having;
they answer different questions and should not be confused for one another.

## Search windows

Search within the expected box dilated by the maximum expected error plus a
margin — never the whole screen, which is how a template matches a similar-looking
element somewhere else entirely and reports a confident, enormous, meaningless
displacement.

A peak landing on the edge of the search window means the element may have moved
further than the window can measure. That is a failure of unknown magnitude
(`missing_or_displaced`), not a pass and not a clamped value.
`LayoutProfile.validate()` refuses a search margin that does not exceed
`tol_fail`, because such a window cannot contain a real failure.

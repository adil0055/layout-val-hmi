# What was measured

Two defaults in this implementation differ from the usual advice for this kind of
system. Both are measured rather than argued, and both are reproducible:

```bash
python benchmarks/estimator_comparison.py
```

## 1. Sub-pixel ZNCC beats phase correlation on cluster content

The usual ranking puts `cv2.phaseCorrelate` as the primary shift estimate
(σ ≈ 0.05–0.2 px) and `cv2.matchTemplate` as identity confirmation
(σ ≈ 0.1–0.3 px). Measured against a *known injected shift* on simulated cluster
content, the ordering reverses, and it does so in every imaging regime tried:

| Regime | ZNCC RMS | ZNCC max | Phase RMS | Phase max |
|---|---|---|---|---|
| baseline (sampling ratio 1.6) | 0.107 px | 0.343 px | 0.263 px | 0.707 px |
| noisy (σ = 6.0 grey levels) | 0.107 px | 0.343 px | 0.230 px | 0.654 px |
| under-sampled (ratio 1.0) | 0.141 px | 0.500 px | 0.282 px | 0.657 px |
| well-sampled (ratio 3.0) | 0.093 px | 0.261 px | 0.292 px | 0.706 px |
| soft focus (defocus σ = 2.0) | 0.199 px | 0.795 px | 0.302 px | 0.521 px |

Errors are radial, in display pixels, over four elements × thirteen shifts.

The reason is content, not implementation. Phase correlation is strong on richly
textured patches where the whole spectrum contributes. A cluster element is a
sparse, high-contrast glyph on a near-uniform background: the normalised
correlation peak is sharp and well conditioned, while phase correlation pays for
the Hann window (which tapers to zero exactly where the glyph's edges are) and for
content entering the patch that was not in it before.

Two things follow in the implementation:

- The element's `context_px` pads the phase-correlation patch **per axis**, so a
  long thin element gets context along its length. Without it, a 120×18 px fuel
  bar has both its only features sitting under the window's taper.
- `Measurement` carries `zncc_dx/dy` and `phase_dx/dy` separately, always, plus
  `estimator_disagreement_px`. The coarse ZNCC peak is within half a pixel of the
  truth by construction, so the two must agree to about that; when they do not,
  the phase estimate is outside its validity regime, `method` records
  `zncc(phase-disagrees)`, and neither number is discarded.

`ElementSpec.estimator` is `auto`, `zncc` or `phase` per element. Do not take
either default on faith — measure it on your own content and rig.

## 2. A repeatability study cannot set a tolerance on its own

The standard rule is: lock the rig, show a static screen, take ~200 frames over
~30 minutes, report σ per element, and set `tol_fail ≥ 3σ + spec_tolerance`.

That is the right study and the wrong stopping point. It measures **noise**.
It does not measure **accuracy**, and the gap between the two is not small.

Correlation-based sub-pixel estimators suffer *peak locking*: the estimate is
pulled towards particular fractions of a pixel, so the error is a function of the
true displacement's fractional part. On a static screen that fractional part is
always zero. The bias is exactly invisible to the study that is supposed to
qualify the measurement.

Measured on the simulator, one element, one run:

```
FUEL_BAR   repeatability sigma = 0.005 px      (static screen, 12 frames)
FUEL_BAR   error vs known shift = 0.28 px      (same element, same run)
```

`3σ + spec` from the repeatability study alone would have justified a tolerance
roughly twenty times tighter than the chain can honour. The sawtooth is plain
once you look for it — error near zero at fractional 0.00 and 0.50, near −0.25 px
at 0.25 and 0.75:

```
shift    zncc err
 0.00      +0.025
 0.25      -0.228
 0.50      -0.031
 0.75      -0.280
 1.00      +0.017
 1.25      -0.232
```

So `layoutval.linearity` exists alongside `layoutval.repeatability`, and the
defensible floor is:

```
tol_fail >= 3*sigma + subpixel_bias + spec_tolerance
```

`Tolerance.defensible_floor()` computes it, `gate()` enforces it, and
`LayoutProfile.validate()` complains when a profile asks for less. `gate()` also
complains when σ is implausibly small and no linearity study has been applied,
because that combination is how the mistake gets made.

### What the linearity study does and does not cover

`run_linearity` shifts a captured reference by known fractional amounts and
measures what comes back. That characterises the estimator's interpolation bias
on your actual imagery — which is the dominant term when the sampling ratio is
low — but it warps an already-captured frame, so it does not reproduce the camera
re-sampling a genuinely moved element. **It is a floor on the real error, not the
whole of it.** The study records that caveat in its own metadata.

Where the build can be made to cooperate, prefer the real thing:
`linearity_from_observations` takes `(commanded_dx, commanded_dy, measured_dx,
measured_dy)` rows gathered by having the HMI render an element at known offsets,
or by driving a continuously-variable element to known values. That path includes
the optics and supersedes the synthetic one.

## 3. Mask centroids are repeatable and not accurate

The usual advice measures an isolated luminous telltale by its mask centroid
(σ ≈ 0.05–0.2 px). On repeatability grounds that looks right — a mask centroid is
very stable frame to frame. Against a known displacement it is not: thresholding
re-quantises the anti-aliased edge differently at every sub-pixel phase.

```
TELLTALE_BATTERY_LOW   sigma 0.151 px      linearity max error 1.593 px
TELLTALE_OIL_PRESSURE  sigma 0.121 px      linearity max error 0.471 px
```

So `measure_telltale` reports position from correlation and uses the mask for
what only the mask can tell you — whether the element is lit at all
(`Measurement.element_absent`, reported as its own verdict reason) and how much of
it there is (`area_px`). The centroid is still computed and recorded in
`centroid_dx/dy`, and `estimator="centroid"` selects it. Run the linearity study
before doing so.

## Reproducing

`layoutval demo` runs the whole build order and finishes with a ground-truth
check, so a regression in any stage shows up as a number that no longer matches
what was injected. `tests/` covers the same ground with assertions.

The simulator is a test fixture, not a model of any camera. Numbers measured
against it say the code is correct. What a real rig can resolve is what the
repeatability and linearity studies on that rig say, and nothing else.

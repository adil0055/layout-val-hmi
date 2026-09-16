"""The sub-pixel linearity study -- what a repeatability study cannot see.

A repeatability study takes many frames of a **static** screen.  Nothing moves,
so every ``dx, dy`` it reports is measurement noise, and its sigma is the
right number for "how much does this reading wobble".

It is the wrong number for "how accurately can this rig measure a displacement",
and the difference is not academic.  Correlation-based sub-pixel estimators
suffer from *peak locking*: the estimate is pulled towards particular fractions
of a pixel, so the error is a function of the true displacement's fractional
part.  On a static screen that fractional part is always zero, so the bias never
shows up.  This implementation's own simulator measures sigma = 0.005 px on a
static screen and a 0.28 px systematic error against a known injected shift, on
the same element, in the same run.  Taking ``tol_fail >= 3*sigma + spec`` from
the repeatability study alone would have set a tolerance roughly twenty times
tighter than the chain can actually honour.

So: characterise the sub-pixel response too, by shifting a captured reference by
known fractional amounts and measuring what comes back.

**What this does and does not cover.**  Warping an already-captured frame does
not reproduce the camera re-sampling a genuinely moved element, so the number it
returns is a floor, not the whole error.  It captures the estimator's
interpolation bias, which is the dominant term when the sampling ratio is low.
Where the HMI can render an element at a commanded offset -- or where a
continuously driven element like a fuel bar exists -- prefer measuring those,
via :func:`linearity_from_observations`, because that path includes the optics.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from layoutval.measure import MaskSpec, measure_element
from layoutval.profile import LayoutProfile
from layoutval.types import resolve_value

DEFAULT_SHIFTS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate by a fractional amount with cubic interpolation."""
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


@dataclass
class ElementLinearity:
    """Sub-pixel response of one element."""

    element_id: str
    n: int
    rms_error_px: float
    max_error_px: float
    """The number that belongs in a tolerance: the worst systematic error the
    estimator makes as a function of where between two pixels the element lands."""

    mean_error_px: float
    errors: list[tuple[float, float, float]] = field(default_factory=list)
    """``(commanded_dx, commanded_dy, error_px)`` for every probe, so the
    sawtooth can be plotted rather than taken on trust."""

    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LinearityStudy:
    screen: str
    method: str
    elements: dict[str, ElementLinearity] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen": self.screen,
            "method": self.method,
            "metadata": self.metadata,
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "LinearityStudy":
        d = json.loads(Path(path).read_text())
        elements = {
            k: ElementLinearity(
                **{**v, "errors": [tuple(e) for e in v.get("errors", [])]}
            )
            for k, v in d["elements"].items()
        }
        return cls(
            screen=d["screen"],
            method=d.get("method", "synthetic_shift"),
            elements=elements,
            metadata=d.get("metadata", {}),
        )


def run_linearity(
    profile: LayoutProfile,
    reference: np.ndarray,
    *,
    shifts: Sequence[float] = DEFAULT_SHIFTS,
    values: dict[str, float] | None = None,
    mask_spec: MaskSpec | None = None,
    diagonal: bool = True,
) -> LinearityStudy:
    """Shift a captured reference by known fractional amounts and measure it.

    Each probe displaces the whole frame, so every element is exercised at every
    sub-pixel phase at once.  ``diagonal`` applies the shift to both axes, which
    is the case a single-axis probe would miss.
    """
    per: dict[str, list[tuple[float, float, float]]] = {s.id: [] for s in profile}
    failures: dict[str, int] = {s.id: 0 for s in profile}

    for s in shifts:
        dx, dy = (float(s), float(s) if diagonal else 0.0)
        probe = shift_image(reference, dx, dy)
        for spec in profile:
            value = resolve_value(spec, values)
            m = measure_element(
                reference,
                probe,
                spec,
                value=value,
                template=profile.template(spec.id),
                mask_spec=MaskSpec.from_config(spec.mask) if spec.mask else mask_spec,
            )
            if m.dx is None or m.dy is None:
                failures[spec.id] += 1
                continue
            per[spec.id].append((dx, dy, float(math.hypot(m.dx - dx, m.dy - dy))))

    elements: dict[str, ElementLinearity] = {}
    for eid, rows in per.items():
        if not rows:
            elements[eid] = ElementLinearity(
                eid, 0, float("nan"), float("nan"), float("nan"), [], failures[eid]
            )
            continue
        errs = np.array([r[2] for r in rows], dtype=np.float64)
        elements[eid] = ElementLinearity(
            element_id=eid,
            n=len(rows),
            rms_error_px=float(np.sqrt((errs**2).mean())),
            max_error_px=float(errs.max()),
            mean_error_px=float(errs.mean()),
            errors=[(float(a), float(b), float(c)) for a, b, c in rows],
            failures=failures[eid],
        )
    return LinearityStudy(
        screen=profile.screen,
        method="synthetic_shift",
        elements=elements,
        metadata={
            "shifts": list(shifts),
            "diagonal": diagonal,
            "caveat": (
                "synthetic warp of a captured frame; a floor on the real error, "
                "because it does not include the camera re-sampling a moved element"
            ),
        },
    )


def linearity_from_observations(
    observations: dict[str, Sequence[tuple[float, float, float, float]]],
    *,
    screen: str,
    method: str = "commanded_shift",
) -> LinearityStudy:
    """Build a study from real commanded displacements.

    ``observations`` maps element id to ``(commanded_dx, commanded_dy,
    measured_dx, measured_dy)`` rows, gathered by having the HMI render an
    element at known offsets or by driving a moving element to known values.
    This path includes the optics, so it supersedes the synthetic one wherever
    the build can be made to cooperate.
    """
    elements: dict[str, ElementLinearity] = {}
    for eid, rows in observations.items():
        errs = np.array(
            [math.hypot(mx - cx, my - cy) for cx, cy, mx, my in rows], dtype=np.float64
        )
        if not len(errs):
            continue
        elements[eid] = ElementLinearity(
            element_id=eid,
            n=len(errs),
            rms_error_px=float(np.sqrt((errs**2).mean())),
            max_error_px=float(errs.max()),
            mean_error_px=float(errs.mean()),
            errors=[(float(c[0]), float(c[1]), float(e)) for c, e in zip(rows, errs)],
        )
    return LinearityStudy(screen=screen, method=method, elements=elements)


def apply_linearity(profile: LayoutProfile, study: LinearityStudy) -> list[str]:
    """Record each element's sub-pixel bias in its tolerance.

    Returns the ids the study did not cover.
    """
    missing: list[str] = []
    for spec in profile:
        lin = study.elements.get(spec.id)
        if lin is None or not math.isfinite(lin.max_error_px):
            missing.append(spec.id)
            continue
        spec.tolerance = replace(spec.tolerance, subpixel_bias_px=lin.max_error_px)
    return missing


def best_estimator(
    profile: LayoutProfile,
    reference: np.ndarray,
    *,
    shifts: Sequence[float] = DEFAULT_SHIFTS,
    values: dict[str, float] | None = None,
    margin: float = 1.15,
) -> dict[str, str]:
    """Pick each element's estimator by measured sub-pixel accuracy.

    This is the stronger criterion than the repeatability study's sigma, because
    it compares the estimators against a known displacement rather than against
    their own wobble.  Returns the chosen estimator per element; elements where
    neither is clearly better are left alone.
    """
    original = {spec.id: spec.estimator for spec in profile}
    candidates = ("zncc", "phase", "centroid")
    scores: dict[str, dict[str, float]] = {}
    try:
        for candidate in candidates:
            for spec in profile:
                spec.estimator = candidate
            study = run_linearity(profile, reference, shifts=shifts, values=values)
            for eid, lin in study.elements.items():
                if math.isfinite(lin.rms_error_px):
                    scores.setdefault(eid, {})[candidate] = lin.rms_error_px
    finally:
        for spec in profile:
            spec.estimator = original[spec.id]

    chosen: dict[str, str] = {}
    for eid, s in scores.items():
        # "centroid" is a no-op for kinds that do not implement it, so it scores
        # identically to the default there and the margin keeps it from winning.
        ranked = sorted(s.items(), key=lambda kv: kv[1])
        if len(ranked) < 2:
            continue
        (best, best_err), (_, next_err) = ranked[0], ranked[1]
        if best_err > 0 and next_err > best_err * margin:
            chosen[eid] = best
    return chosen

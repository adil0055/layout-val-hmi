"""Core data model.

Every geometric quantity in this package is expressed in **display pixels**,
in display space, after rectification.  That is deliberate: it makes a
tolerance something the HMI team can argue about ("the icon may not move more
than 2 px") rather than something only the rig owner can interpret ("no more
than 3.4 camera px at the current standoff").
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class Verdict(str, Enum):
    """Outcome of a single check.  Ordered worst-first for aggregation."""

    FAIL = "FAIL"
    REVIEW = "REVIEW"
    PASS = "PASS"

    @property
    def rank(self) -> int:
        return {"FAIL": 0, "REVIEW": 1, "PASS": 2}[self.value]

    @staticmethod
    def worst(verdicts: Sequence["Verdict"]) -> "Verdict":
        if not verdicts:
            return Verdict.PASS
        return min(verdicts, key=lambda v: v.rank)


class ElementKind(str, Enum):
    """What an element *is*, which decides how it is measured.

    The distinction is not cosmetic.  A needle rotates about a pivot and is not
    a shift-invariant patch, so phase correlation returns nonsense for it; a
    luminous telltale on a dark background is measured far better by its mask
    centroid than by correlation.  Picking the wrong kind is the most common way
    to get a confidently wrong number out of this pipeline.
    """

    ICON = "icon"
    """Symbol / telltale glyph.  Translation-only; phase correlation + ZNCC."""

    TEXT = "text"
    """Text block.  Translation-only, but always measured on luma -- sub-pixel
    anti-aliasing puts colour fringes on glyph edges whose weight depends on
    sub-pixel phase, which shifts the apparent per-channel centroid."""

    TELLTALE = "telltale"
    """Isolated luminous element.  Mask centroid (reuses the HSV masks from
    colour/symbol validation) with ZNCC for identity."""

    NEEDLE = "needle"
    """Rotating indicator.  Pivot position and angle are validated as two
    separate quantities; a needle at the right angle about the wrong pivot is a
    real and separate defect."""

    REGION = "region"
    """A generic textured widget measured by correlation.  Also the fallback."""


@dataclass(frozen=True)
class Tolerance:
    """Per-element acceptance limits, all in display pixels (degrees for angle).

    ``tol_fail`` must be defensible: the repeatability study (see
    :mod:`layoutval.repeatability`) gives you 3-sigma for this element, and the
    rule is ``tol_fail >= 3*sigma + spec_tolerance``.  ``sigma_px`` carries the
    measured noise forward so :meth:`check_against_sigma` can say out loud when
    a tolerance is tighter than the rig can actually resolve.
    """

    tol_warn: float = 1.5
    tol_fail: float = 2.5
    identity_min: float = 0.80
    """Minimum ZNCC for 'this is still the same thing'.  Below it the element is
    reported as wrong content, not as a position failure."""

    angle_warn_deg: float = 1.0
    angle_fail_deg: float = 2.0
    sigma_px: float | None = None
    """Measured 1-sigma repeatability, filled in by the repeatability study."""

    subpixel_bias_px: float = 0.0
    """Worst systematic error the estimator makes as a function of where between
    two pixels the element lands, from the linearity study.  A static-screen
    repeatability study cannot see this, and leaving it out of the arithmetic is
    how a rig ends up with a tolerance far tighter than it can honour."""

    spec_tolerance_px: float = 0.0
    """The tolerance the requirement actually asks for, before measurement
    noise is added.  Kept separately so the arithmetic stays auditable."""

    def check_against_sigma(self) -> str | None:
        """Return a human-readable complaint if ``tol_fail`` is not defensible.

        Returns ``None`` when the tolerance is supported by the measured noise.
        A rig that cannot resolve the requirement should be reported as such and
        fixed, not shipped: a suite that fails randomly gets marked flaky and is
        ignored inside two months, which is worse than not having the test.
        """
        if self.sigma_px is None:
            return None
        floor = self.defensible_floor()
        if self.tol_fail < floor:
            return (
                f"tol_fail={self.tol_fail:.2f} px is below the defensible floor "
                f"3*sigma + bias + spec = 3*{self.sigma_px:.3f} + "
                f"{self.subpixel_bias_px:.3f} + {self.spec_tolerance_px:.2f} = "
                f"{floor:.2f} px"
            )
        return None

    def defensible_floor(self) -> float:
        """``3*sigma + subpixel_bias + spec_tolerance``.

        Noise, systematic sub-pixel error, and what the requirement actually
        asks for.  Dropping the middle term is the common mistake -- it is
        invisible to a static-screen study and it is often the largest of the
        three.
        """
        sigma = self.sigma_px or 0.0
        return 3.0 * sigma + self.subpixel_bias_px + self.spec_tolerance_px


@dataclass(frozen=True)
class PositionModel:
    """Where an element is *expected* to be, as a function of signal state.

    Progress bars, fuel bars and scrolling lists are supposed to move.  Their
    expected position is a function of state, not a constant; modelling them as
    constants means either testing nothing (tolerance wide enough to cover the
    travel) or failing constantly.
    """

    kind: str = "static"
    """``static`` or ``linear``."""

    origin: tuple[float, float] = (0.0, 0.0)
    """Expected top-left of the element at ``value == value_min``."""

    direction: tuple[float, float] = (0.0, 0.0)
    """Travel in display px across the full ``value_min..value_max`` range."""

    value_min: float = 0.0
    value_max: float = 1.0

    def expected_top_left(self, value: float | None = None) -> tuple[float, float]:
        if self.kind == "static" or value is None:
            return self.origin
        if self.kind != "linear":
            raise ValueError(f"unknown position model kind: {self.kind!r}")
        span = self.value_max - self.value_min
        if span == 0:
            t = 0.0
        else:
            t = (value - self.value_min) / span
        t = min(max(t, 0.0), 1.0)
        return (
            self.origin[0] + self.direction[0] * t,
            self.origin[1] + self.direction[1] * t,
        )


@dataclass(frozen=True)
class AngleModel:
    """Expected needle angle as a function of signal value.

    The same argument as :class:`PositionModel`: a gauge needle's expected angle
    is a function of state, not a constant.  Angles are degrees from the +x axis,
    counter-clockwise positive as seen on the display.
    """

    kind: str = "static"
    """``static`` or ``linear``."""

    angle_at_min: float = 0.0
    angle_at_max: float = 0.0
    value_min: float = 0.0
    value_max: float = 1.0

    def context(self) -> tuple[int, int]:
        """Resolved context margin in display pixels, as ``(pad_x, pad_y)``.

        Per axis, because a long thin element needs context along its length to
        measure x and would gain nothing from the same margin above and below.
        """
        if isinstance(self.context_px, (tuple, list)):
            return int(self.context_px[0]), int(self.context_px[1])
        if self.context_px is not None:
            return int(self.context_px), int(self.context_px)
        return (
            int(min(max(round(0.4 * self.bbox[2]), 6), 32)),
            int(min(max(round(0.4 * self.bbox[3]), 6), 32)),
        )

    def anchor_offset(self) -> tuple[float, float]:
        """Where the stored template sits inside the element box."""
        if self.template_anchor is None:
            return (0.0, 0.0)
        return (
            self.template_anchor[0] - self.bbox[0],
            self.template_anchor[1] - self.bbox[1],
        )

    def template_origin(self, value: float | None = None) -> tuple[float, float]:
        """Expected display position of the stored template's top-left."""
        x, y = self.position.expected_top_left(value)
        ox, oy = self.anchor_offset()
        return (x + ox, y + oy)

    def expected_angle(self, value: float | None = None) -> float | None:
        if self.kind == "static":
            return self.angle_at_min
        if self.kind != "linear":
            raise ValueError(f"unknown angle model kind: {self.kind!r}")
        if value is None:
            return None
        span = self.value_max - self.value_min
        t = 0.0 if span == 0 else (value - self.value_min) / span
        t = min(max(t, 0.0), 1.0)
        return self.angle_at_min + (self.angle_at_max - self.angle_at_min) * t


@dataclass
class ElementSpec:
    """One element of the layout inventory, in display coordinates.

    ``source`` records where the expected geometry came from -- ``design`` (an
    export from the HMI design tool), ``teachin`` (differential teach-in against
    a CAN signal) or ``manual`` (assisted annotation).  It is carried into the
    report because it is exactly what decides what a green result means: a
    design-sourced expectation answers "does the build match the design", a
    teach-in expectation only answers "does the build match the last build".
    """

    id: str
    kind: ElementKind = ElementKind.ICON
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """(x, y, w, h) in display pixels, at the element's nominal state."""

    position: PositionModel = field(default_factory=PositionModel)
    tolerance: Tolerance = field(default_factory=Tolerance)
    source: str = "teachin"
    signal: str | None = None
    """CAN signal that toggles or drives the element, when there is one."""

    theme: str | None = None
    template_path: str | None = None
    """Reference patch on disk, relative to the profile file."""

    template_anchor: tuple[float, float] | None = None
    """Display coordinate the stored template's top-left corresponds to, at the
    nominal state.  Defaults to the bbox origin.

    It is separate from :attr:`bbox` because the two answer different questions.
    A design export gives a *layout* box, padding included; a template taught by
    toggling a signal is cut to the *ink*, which normally sits somewhere inside
    that box.  Differencing a position measured against one anchor with an
    expectation expressed in the other reports the padding as a defect, on every
    run, for every element -- with a magnitude that looks entirely plausible."""

    pivot: tuple[float, float] | None = None
    """Needle pivot in display coordinates."""

    expected_angle_deg: float | None = None
    """Needle angle at the nominal state, degrees from the +x axis,
    counter-clockwise positive in display coordinates.  Use :attr:`angle` instead
    when the needle is driven by a signal."""

    angle: AngleModel | None = None
    """Expected angle as a function of signal value, for a driven needle."""

    search_margin_px: float = 8.0
    """How far outside the expected box to search.  Must exceed the largest
    displacement you intend to be able to *measure*; a peak landing on the edge
    of this window is reported as a failure of unknown magnitude, never a pass
    and never a clamped value."""

    estimator: str = "auto"
    """Which shift estimate is reported: ``zncc``, ``phase`` or ``auto``.

    Both are always computed and both are always in the record -- see
    :attr:`Measurement.zncc_dx` and :attr:`Measurement.phase_dx`.  This only
    decides which one becomes ``dx``/``dy``.

    ``auto`` reports the ZNCC estimate and treats phase correlation as a
    cross-check.  That is the opposite of the usual advice, and it is what this
    implementation's own benchmark measures on cluster content: sparse
    high-contrast glyphs on a near-uniform background give a sharp, well
    conditioned correlation peak, while phase correlation pays for the Hann
    window and for content entering the patch.  Run ``benchmarks/estimator_
    comparison.py``, or let the repeatability study choose per element from
    measured sigma -- do not take either default on faith."""

    context_px: int | tuple[int, int] | None = None
    """Context to include around the element for the phase-correlation stage.

    A Hann window tapers to zero at the patch edge, so a template cut tightly
    around an element puts that element's edges -- its only features -- exactly
    where the window suppresses them.  ``None`` picks a margin from the element's
    size; raise it for a large plain element, lower it where a neighbour would be
    dragged into the patch."""

    mask: dict[str, Any] | None = None
    """How to binarise this element for mask-based measurement (telltales and
    needles).  ``None`` uses the run-wide default.  A gauge needle sharing a
    region with a desaturated plate needs a hue-and-saturation mask, not a luma
    threshold -- reuse the HSV masks already built for colour validation."""

    occludes: list[str] = field(default_factory=list)
    """Element ids this one is required to be drawn on top of.  Z-order is
    invisible to per-element position checks: two elements each in the right
    place with the wrong one on top passes everything else you can write."""

    notes: str = ""

    def __post_init__(self) -> None:
        # A spec built with a bbox but no position model means "static, here".
        # Leaving the model's origin at (0, 0) would put every such element's
        # expected box in the top-left corner of the display.
        if (
            self.position.kind == "static"
            and tuple(self.position.origin) == (0.0, 0.0)
            and tuple(self.bbox[:2]) != (0.0, 0.0)
        ):
            self.position = PositionModel(origin=(float(self.bbox[0]), float(self.bbox[1])))

    def expected_bbox(self, value: float | None = None) -> tuple[float, float, float, float]:
        x, y = self.position.expected_top_left(value)
        return (x, y, self.bbox[2], self.bbox[3])

    def expected_centre(self, value: float | None = None) -> tuple[float, float]:
        x, y, w, h = self.expected_bbox(value)
        return (x + w / 2.0, y + h / 2.0)

    def context(self) -> tuple[int, int]:
        """Resolved context margin in display pixels, as ``(pad_x, pad_y)``.

        Per axis, because a long thin element needs context along its length to
        measure x and would gain nothing from the same margin above and below.
        """
        if isinstance(self.context_px, (tuple, list)):
            return int(self.context_px[0]), int(self.context_px[1])
        if self.context_px is not None:
            return int(self.context_px), int(self.context_px)
        return (
            int(min(max(round(0.4 * self.bbox[2]), 6), 32)),
            int(min(max(round(0.4 * self.bbox[3]), 6), 32)),
        )

    def anchor_offset(self) -> tuple[float, float]:
        """Where the stored template sits inside the element box."""
        if self.template_anchor is None:
            return (0.0, 0.0)
        return (
            self.template_anchor[0] - self.bbox[0],
            self.template_anchor[1] - self.bbox[1],
        )

    def template_origin(self, value: float | None = None) -> tuple[float, float]:
        """Expected display position of the stored template's top-left."""
        x, y = self.position.expected_top_left(value)
        ox, oy = self.anchor_offset()
        return (x + ox, y + oy)

    def expected_angle(self, value: float | None = None) -> float | None:
        """The modelled angle for this state, falling back to the fixed one."""
        if self.angle is not None:
            a = self.angle.expected_angle(value)
            if a is not None:
                return a
        return self.expected_angle_deg


def resolve_value(spec: ElementSpec, values: Mapping[str, float] | None) -> float | None:
    """Look up the signal value that decides where ``spec`` is expected to be.

    Callers key the mapping by element id or by signal name, whichever they have
    to hand.  Resolving it in one place keeps the measurement, the residual check
    and the overlay all asking about the same expected box -- when they disagree,
    the report contradicts its own picture.
    """
    if not values:
        return None
    if spec.id in values:
        return values[spec.id]
    if spec.signal and spec.signal in values:
        return values[spec.signal]
    return None


@dataclass
class Measurement:
    """Raw numbers out of the measurement stage, before any verdict is formed."""

    element_id: str
    dx: float | None = None
    dy: float | None = None
    zncc: float | None = None
    phase_response: float | None = None
    method: str = ""
    zncc_dx: float | None = None
    zncc_dy: float | None = None
    """The coarse ZNCC estimate, kept even when phase correlation refined it."""

    phase_dx: float | None = None
    phase_dy: float | None = None
    """The phase-correlation estimate, kept even when it was rejected."""

    centroid_dx: float | None = None
    centroid_dy: float | None = None
    """Mask-centroid displacement, recorded for every telltale whether or not it
    is the reported estimate."""

    element_absent: bool = False
    """The mask is empty where the reference had one.  Reported as its own
    reason, because "not drawn at all" and "drawn wrongly" send a defect to
    different people."""

    estimator_disagreement_px: float | None = None
    """Distance between the two shift estimates.  Large means one of them is
    outside its validity regime, and the record says which one was used."""

    observed_centre: tuple[float, float] | None = None
    expected_centre: tuple[float, float] | None = None
    peak_on_search_border: bool = False
    """The correlation peak sat on the edge of the search window.  The element
    may have moved further than the window can measure, so the magnitude is
    unknown."""

    angle_deg: float | None = None
    expected_angle_deg: float | None = None
    d_angle_deg: float | None = None
    pivot_observed: tuple[float, float] | None = None
    area_px: float | None = None
    error: str | None = None

    @property
    def abs_delta(self) -> float | None:
        if self.dx is None or self.dy is None:
            return None
        return math.hypot(self.dx, self.dy)


@dataclass
class ElementResult:
    """A measurement plus the verdict formed from it."""

    element_id: str
    verdict: Verdict
    reason: str | None
    measurement: Measurement
    tolerance: Tolerance
    source: str = "teachin"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "element_id": self.element_id,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "source": self.source,
            "measurement": asdict(self.measurement),
            "tolerance": asdict(self.tolerance),
        }
        d["measurement"]["abs_delta"] = self.measurement.abs_delta
        return d


@dataclass
class ResidualFinding:
    """An unmodelled difference found by the whole-frame residual check.

    Always advisory.  Pixel-level comparison of camera captures is noisy enough
    that a hard threshold either fires constantly or is set so loose it catches
    nothing; the job of this check is to surface the stray artefact, the element
    nobody taught and the wrong z-order, not to issue verdicts.
    """

    bbox: tuple[int, int, int, int]
    mean_dissimilarity: float
    area_px: int
    overlaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunReport:
    """Everything one validation run produced."""

    screen: str
    theme: str | None = None
    results: list[ElementResult] = field(default_factory=list)
    residual_score: float | None = None
    residual_findings: list[ResidualFinding] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    """Run-level conditions that are not per-element verdicts: rig drift,
    settling timeouts, tolerances that the repeatability study does not
    support."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def flag(self, name: str, **details: Any) -> None:
        self.flags.append({"flag": name, **details})

    @property
    def verdict(self) -> Verdict:
        v = Verdict.worst([r.verdict for r in self.results])
        if any(f.get("severity") == "fail" for f in self.flags):
            return Verdict.FAIL
        if v is Verdict.PASS and (self.flags or self.residual_findings):
            return Verdict.REVIEW
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen": self.screen,
            "theme": self.theme,
            "verdict": self.verdict.value,
            "metadata": self.metadata,
            "flags": self.flags,
            "residual": {
                "ssim": self.residual_score,
                "findings": [f.to_dict() for f in self.residual_findings],
            },
            "elements": [r.to_dict() for r in self.results],
            "summary": {
                v.value: sum(1 for r in self.results if r.verdict is v) for v in Verdict
            },
        }

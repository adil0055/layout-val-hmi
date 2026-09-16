"""The repeatability study, and the gate it creates.

Do this before setting a single tolerance value.  It is the difference between a
script and a qualified measurement system, and it is what an ASPICE or
ISO 26262 reviewer will ask for.

1. Lock the rig.  Display one static screen.
2. Capture ~200 frames spread over ~30 minutes, so warm-up and thermal drift are
   in the data and not just sensor noise.
3. Run the full pipeline on every frame.  Nothing moved, so every ``dx, dy``
   reported is your own measurement noise.
4. Report sigma per element.  Elements differ -- a large high-contrast icon
   measures far more repeatably than three pixels of thin text.
5. Practical resolution is 3*sigma.  Set ``tol_fail >= 3*sigma + spec_tolerance``.
6. Tear the rig down, rebuild it, repeat.  That gives reproducibility as well as
   repeatability.

The result you must be willing to report: if 3*sigma comes out at 1.2 px and the
requirement is +/-1 px, **this rig cannot test that requirement**.  Say so and
fix the rig.  Shipping the test anyway produces a suite that fails randomly,
gets marked flaky, and is ignored inside two months -- which is worse than not
having the test.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from layoutval.capture import FrameSource
from layoutval.pipeline import Pipeline
from layoutval.profile import LayoutProfile
from layoutval.types import Measurement


@dataclass
class ElementNoise:
    """Measurement noise for one element, in display pixels."""

    element_id: str
    n: int
    sigma_x: float
    sigma_y: float
    bias_x: float
    bias_y: float
    max_abs_delta: float
    sigma_angle_deg: float | None = None
    sigma_zncc: float | None = None
    sigma_phase: float | None = None
    """Radial sigma of each shift estimator taken on its own.  A static screen
    makes these directly comparable, so the study -- not a default -- can decide
    which estimator an element should use."""

    failures: int = 0
    """Frames on which this element could not be measured at all.  A non-zero
    count on a static screen is itself a finding."""

    def better_estimator(self, *, margin: float = 1.15) -> str | None:
        """Which estimator measured more repeatably, or ``None`` if it is a draw.

        ``margin`` keeps the study from flipping an element back and forth on a
        difference that is not real; one has to be clearly better to win.
        """
        z, p = self.sigma_zncc, self.sigma_phase
        if z is None or p is None or not (math.isfinite(z) and math.isfinite(p)):
            return None
        if z <= 0 or p <= 0:
            return None
        if p * margin < z:
            return "phase"
        if z * margin < p:
            return "zncc"
        return None

    @property
    def sigma(self) -> float:
        """Radial 1-sigma.

        ``sqrt(sigma_x^2 + sigma_y^2)``: the tolerance is applied to ``|delta|``,
        so the noise it must be compared against is the noise of ``|delta|``, not
        of either axis alone.
        """
        return math.hypot(self.sigma_x, self.sigma_y)

    @property
    def resolution(self) -> float:
        """Practical resolution, 3-sigma."""
        return 3.0 * self.sigma

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sigma"] = self.sigma
        d["resolution_3sigma"] = self.resolution
        return d


@dataclass
class RepeatabilityStudy:
    """The per-element noise report, plus the conditions it was taken under."""

    screen: str
    frames: int
    duration_s: float
    elements: dict[str, ElementNoise] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen": self.screen,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "metadata": self.metadata,
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "RepeatabilityStudy":
        d = json.loads(Path(path).read_text())
        elements = {}
        for k, v in d["elements"].items():
            v = {kk: vv for kk, vv in v.items() if kk not in ("sigma", "resolution_3sigma")}
            elements[k] = ElementNoise(**v)
        return cls(
            screen=d["screen"],
            frames=int(d["frames"]),
            duration_s=float(d["duration_s"]),
            elements=elements,
            metadata=d.get("metadata", {}),
        )


def _radial_sigma(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return math.hypot(float(np.std(xs, ddof=1)), float(np.std(ys, ddof=1)))


def study_from_measurements(
    screen: str,
    per_element: dict[str, list[Measurement]],
    *,
    duration_s: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> RepeatabilityStudy:
    """Build a study from the measurements taken on a static screen.

    Nothing moved, so every ``dx, dy`` in here is the measurement chain's own
    noise.  Both estimators are summarised separately, because which one is
    better is an element-by-element question and this is the data that answers
    it on the rig you actually have.
    """
    elements: dict[str, ElementNoise] = {}
    n_frames = max((len(v) for v in per_element.values()), default=0)
    for eid, samples in per_element.items():
        good = [m for m in samples if m.dx is not None and m.dy is not None]
        failures = len(samples) - len(good)
        if len(good) < 2:
            elements[eid] = ElementNoise(
                eid, len(good), float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), None, None, None, failures,
            )
            continue
        dxs = np.array([m.dx for m in good], dtype=np.float64)
        dys = np.array([m.dy for m in good], dtype=np.float64)
        angles = [m.d_angle_deg for m in good if m.d_angle_deg is not None]
        elements[eid] = ElementNoise(
            element_id=eid,
            n=len(good),
            # ddof=1: this is a sample standard deviation, and with n in the low
            # hundreds the difference is small but the honesty is free.
            sigma_x=float(np.std(dxs, ddof=1)),
            sigma_y=float(np.std(dys, ddof=1)),
            bias_x=float(np.mean(dxs)),
            bias_y=float(np.mean(dys)),
            max_abs_delta=float(np.max(np.hypot(dxs, dys))),
            sigma_angle_deg=float(np.std(angles, ddof=1)) if len(angles) > 1 else None,
            sigma_zncc=_radial_sigma(
                [m.zncc_dx for m in good if m.zncc_dx is not None],
                [m.zncc_dy for m in good if m.zncc_dy is not None],
            ),
            sigma_phase=_radial_sigma(
                [m.phase_dx for m in good if m.phase_dx is not None],
                [m.phase_dy for m in good if m.phase_dy is not None],
            ),
            failures=failures,
        )
    return RepeatabilityStudy(
        screen=screen,
        frames=n_frames,
        duration_s=duration_s,
        elements=elements,
        metadata=metadata or {},
    )


def run_repeatability(
    pipeline: Pipeline,
    source: FrameSource,
    *,
    n_frames: int = 200,
    spread_over_s: float = 1800.0,
    reference: np.ndarray | None = None,
    values: dict[str, float] | None = None,
    progress: Any = None,
) -> RepeatabilityStudy:
    """Run the study against a live, locked rig showing one static screen.

    ``spread_over_s`` defaults to 30 minutes on purpose.  Two hundred frames
    taken back to back measure sensor noise; two hundred frames taken over half
    an hour measure sensor noise *and* the warm-up drift that will be present in
    every real run.
    """
    reference = pipeline.profile.reference() if reference is None else reference
    interval = spread_over_s / max(n_frames - 1, 1)
    samples: dict[str, list[Measurement]] = {s.id: [] for s in pipeline.profile}
    drifts: list[float] = []

    start = time.monotonic()
    for i in range(n_frames):
        target = start + i * interval
        now = time.monotonic()
        if now < target:
            time.sleep(target - now)
        report = pipeline.measure_frame(
            pipeline.acquire(source), values=values, reference=reference
        )
        if "rig_drift_px" in report.metadata:
            drifts.append(float(report.metadata["rig_drift_px"]))
        for r in report.results:
            if r.element_id in samples:
                samples[r.element_id].append(r.measurement)
        if progress:
            progress(i + 1, n_frames)

    elapsed = time.monotonic() - start
    metadata: dict[str, Any] = {
        "geometry_method": pipeline.calibration.geometry.method,
        "sampling_ratio": round(pipeline.calibration.geometry.sampling_ratio(), 3),
        "rig": pipeline.calibration.rig,
    }
    if drifts:
        metadata["rig_drift_px_max"] = round(max(drifts), 3)
    return study_from_measurements(
        pipeline.profile.screen, samples, duration_s=elapsed, metadata=metadata
    )


def apply_study(
    profile: LayoutProfile,
    study: RepeatabilityStudy,
    *,
    set_tolerances: bool = False,
    choose_estimator: bool = True,
) -> list[str]:
    """Write the measured sigma into the profile's tolerances.

    With ``set_tolerances=True`` the limits are *derived* from the study:
    ``tol_fail = 3*sigma + spec_tolerance`` and ``tol_warn`` halfway between the
    spec tolerance and that.  Otherwise only ``sigma_px`` is recorded, so
    :meth:`Tolerance.check_against_sigma` can complain about limits a human set
    that the rig cannot support.

    Returns the list of elements the study did not cover.
    """
    missing: list[str] = []
    for spec in profile:
        noise = study.elements.get(spec.id)
        if noise is None or not math.isfinite(noise.sigma):
            missing.append(spec.id)
            continue
        tol = replace(spec.tolerance, sigma_px=noise.sigma)
        if set_tolerances:
            fail = 3.0 * noise.sigma + tol.subpixel_bias_px + tol.spec_tolerance_px
            tol = replace(
                tol,
                tol_fail=round(fail, 3),
                tol_warn=round(tol.spec_tolerance_px + (fail - tol.spec_tolerance_px) / 2.0, 3),
            )
        spec.tolerance = tol
        if choose_estimator:
            # Which estimator to trust is a per-element, per-rig question, and
            # this is the only data that answers it without a known injected
            # shift.  A draw leaves the element on the default.
            better = noise.better_estimator()
            if better:
                spec.estimator = better
        # The search window has to be able to contain a real failure, otherwise
        # the peak lands on its border and the magnitude comes back unknown.
        spec.search_margin_px = max(spec.search_margin_px, 2.0 * tol.tol_fail + 4.0)
    return missing


@dataclass
class GateResult:
    """Whether the rig can support the tolerances the profile asks for."""

    passed: bool
    problems: list[str] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["repeatability gate: " + ("PASS" if self.passed else "FAIL")]
        lines += [f"  - {p}" for p in self.problems]
        if self.uncovered:
            lines.append(f"  - not covered by the study: {', '.join(self.uncovered)}")
        return "\n".join(lines)


def gate(
    profile: LayoutProfile,
    study: RepeatabilityStudy,
    *,
    bias_floor_px: float = 0.25,
) -> GateResult:
    """Week 2's gate.  Do not build past a failing one.

    Everything after rectification and this study is straightforward
    engineering; everything before it determines whether the numbers those weeks
    produce mean anything.
    """
    problems: list[str] = []
    uncovered: list[str] = []
    for spec in profile:
        noise = study.elements.get(spec.id)
        if noise is None or not math.isfinite(noise.sigma):
            uncovered.append(spec.id)
            continue
        if noise.failures:
            problems.append(
                f"{spec.id}: {noise.failures} of {noise.failures + noise.n} frames could "
                "not be measured on a static screen"
            )
        floor = (
            noise.resolution
            + spec.tolerance.subpixel_bias_px
            + spec.tolerance.spec_tolerance_px
        )
        if spec.tolerance.tol_fail < floor:
            problems.append(
                f"{spec.id}: requirement asks for +/-"
                f"{spec.tolerance.spec_tolerance_px:.2f} px with tol_fail "
                f"{spec.tolerance.tol_fail:.2f} px, but this rig resolves only "
                f"{noise.resolution:.2f} px (3*sigma) plus "
                f"{spec.tolerance.subpixel_bias_px:.2f} px of sub-pixel bias. The rig "
                "cannot test this requirement -- fix the rig (longer lens, shorter "
                "standoff, higher sampling ratio), do not ship the test."
            )
        if spec.tolerance.subpixel_bias_px == 0.0 and noise.sigma < 0.05:
            problems.append(
                f"{spec.id}: sigma is {noise.sigma:.3f} px and no linearity study has "
                "been applied. A static screen cannot reveal sub-pixel bias, so this "
                "sigma alone would justify a tolerance the rig cannot honour -- run "
                "layoutval.linearity before trusting it."
            )
        # The bias floor matters when sigma is very small: 3*sigma alone would
        # flag a quarter-pixel offset that no one can act on and that a real rig
        # would not even resolve.
        bias = math.hypot(noise.bias_x, noise.bias_y)
        if bias > max(3.0 * noise.sigma, bias_floor_px):
            problems.append(
                f"{spec.id}: systematic bias ({noise.bias_x:+.2f}, {noise.bias_y:+.2f}) px "
                "against a static screen -- the reference and the live capture disagree "
                "by more than noise, which points at the geometry, not at the build"
            )
    return GateResult(passed=not problems, problems=problems, uncovered=uncovered)


def reproducibility(studies: Sequence[RepeatabilityStudy]) -> dict[str, dict[str, float]]:
    """Combine studies taken across rig teardowns and rebuilds.

    ``repeatability`` is the pooled within-build sigma; ``reproducibility`` adds
    the spread of the per-build biases, which is the part a teardown exposes and
    a single study cannot see.
    """
    if len(studies) < 2:
        raise ValueError("reproducibility needs at least two studies from separate builds")
    out: dict[str, dict[str, float]] = {}
    ids = set().union(*(set(s.elements) for s in studies))
    for eid in sorted(ids):
        noises = [s.elements[eid] for s in studies if eid in s.elements]
        if len(noises) < 2:
            continue
        within = math.sqrt(float(np.mean([n.sigma**2 for n in noises])))
        biases = np.array([[n.bias_x, n.bias_y] for n in noises], dtype=np.float64)
        between = math.hypot(*np.std(biases, axis=0, ddof=1))
        out[eid] = {
            "repeatability_sigma": within,
            "between_build_sigma": between,
            "reproducibility_sigma": math.hypot(within, between),
            "builds": float(len(noises)),
        }
    return out

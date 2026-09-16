"""The six stages, wired together.

    capture -> undistort -> rectify -> locate -> score -> report

Everything from ``rectify`` onwards is in display pixels.  That is what makes a
tolerance something the HMI team can argue about ("the icon may not move more
than 2 px") instead of something only the rig owner can interpret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from layoutval.calibration import Calibration, DriftTracker, Undistorter
from layoutval.capture import (
    FrameSource,
    SettlingTimeout,
    median_stack,
    capture as capture_frames,
    wait_for_settle,
)
from layoutval.measure import MaskSpec, measure_element
from layoutval.profile import LayoutProfile
from layoutval.residual import check_occlusions, residual_check
from layoutval.types import RunReport, Verdict, resolve_value
from layoutval.verdict import evaluate


@dataclass
class PipelineOptions:
    frames_per_measurement: int = 15
    settle: bool = True
    settle_timeout_s: float = 5.0
    settle_diff_threshold: float = 1.5
    run_residual: bool = True
    residual_threshold: float = 0.45
    residual_min_area_px: int = 64
    mask_spec: MaskSpec = field(default_factory=MaskSpec)


class Pipeline:
    """One configured rig plus one layout profile.

    Construct once per screen and reuse: the undistortion tables and the drift
    reference are built up front rather than per frame.
    """

    def __init__(
        self,
        calibration: Calibration,
        profile: LayoutProfile,
        *,
        options: PipelineOptions | None = None,
        drift_reference: np.ndarray | None = None,
        static_roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.calibration = calibration
        self.profile = profile
        self.options = options or PipelineOptions()
        self.undistorter = (
            Undistorter(calibration.intrinsics) if calibration.intrinsics else None
        )
        self.drift_tracker: DriftTracker | None = None
        if drift_reference is not None and static_roi is not None:
            self.drift_tracker = DriftTracker(
                self.undistort(drift_reference),
                static_roi,
                alarm_px=calibration.drift_alarm_px,
            )

    @classmethod
    def from_files(
        cls,
        calibration_path: Path | str,
        profile_path: Path | str,
        **kwargs: Any,
    ) -> "Pipeline":
        return cls(
            Calibration.load(calibration_path), LayoutProfile.load(profile_path), **kwargs
        )

    # -- stages 1-3 ---------------------------------------------------------

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return self.undistorter(frame) if self.undistorter else frame

    def acquire(self, source: FrameSource, report: RunReport | None = None) -> np.ndarray:
        """Capture -> undistort -> rectify, with the drift check in between.

        A settling timeout fails the case; it does not fall through to measuring
        whatever happened to be on screen when time ran out.
        """
        if self.options.settle:
            try:
                wait_for_settle(
                    source,
                    diff_threshold=self.options.settle_diff_threshold,
                    timeout_s=self.options.settle_timeout_s,
                )
            except SettlingTimeout as exc:
                if report is None:
                    raise
                report.flag("settling_timeout", severity="fail", detail=str(exc))

        raw = median_stack(capture_frames(source, n=self.options.frames_per_measurement))
        undistorted = self.undistort(raw)

        H = self.calibration.geometry.H
        if self.drift_tracker is not None:
            est = self.drift_tracker.measure(undistorted)
            H = self.drift_tracker.corrected_homography(self.calibration.geometry, est)
            if report is not None:
                # Log the magnitude and alarm on it.  Silent compensation is how a
                # rig that someone knocked last Tuesday keeps producing green
                # results for a month.
                report.metadata["rig_drift_px"] = round(est.magnitude_px, 3)
                report.metadata["rig_drift_rotation_deg"] = round(est.rotation_deg, 4)
                if est.exceeds:
                    report.flag(
                        "rig_drift",
                        severity="review",
                        drift_px=round(est.magnitude_px, 3),
                        rotation_deg=round(est.rotation_deg, 4),
                        alarm_px=self.drift_tracker.alarm_px,
                        converged=est.converged,
                        detail=(
                            "rig has moved beyond the alarm threshold; the correction "
                            "was applied but this run should not be trusted until the "
                            "rig is checked"
                        ),
                    )
        return self.calibration.geometry.rectify(undistorted, H)

    # -- stages 4-6 ---------------------------------------------------------

    def measure_frame(
        self,
        live_display: np.ndarray,
        *,
        values: dict[str, float] | None = None,
        reference: np.ndarray | None = None,
        report: RunReport | None = None,
    ) -> RunReport:
        """Locate, score and assemble the record for one rectified frame."""
        reference = self.profile.reference() if reference is None else reference
        report = report or RunReport(screen=self.profile.screen, theme=self.profile.theme)

        for problem in self.profile.validate():
            report.flag("profile", severity="review", detail=problem)

        for spec in self.profile:
            value = resolve_value(spec, values)
            measurement = measure_element(
                reference,
                live_display,
                spec,
                value=value,
                template=self.profile.template(spec.id),
                mask_spec=(
                    MaskSpec.from_config(spec.mask) if spec.mask else self.options.mask_spec
                ),
            )
            report.results.append(evaluate(spec, measurement))

        report.results.extend(
            check_occlusions(reference, live_display, list(self.profile), values=values)
        )

        if self.options.run_residual:
            score, findings = residual_check(
                reference,
                live_display,
                list(self.profile),
                values=values,
                dissimilarity_threshold=self.options.residual_threshold,
                min_area_px=self.options.residual_min_area_px,
            )
            report.residual_score = score
            report.residual_findings = findings

        report.metadata.setdefault("display_size", list(self.profile.display_size))
        report.metadata.setdefault(
            "sampling_ratio", round(self.calibration.geometry.sampling_ratio(), 3)
        )
        report.metadata.setdefault("geometry_method", self.calibration.geometry.method)
        report.metadata.setdefault(
            "expectation_sources",
            sorted({s.source for s in self.profile}),
        )
        return report

    def run(
        self,
        source: FrameSource,
        *,
        values: dict[str, float] | None = None,
        reference: np.ndarray | None = None,
    ) -> RunReport:
        """A complete validation run against a live rig."""
        report = RunReport(screen=self.profile.screen, theme=self.profile.theme)
        live = self.acquire(source, report)
        if any(f.get("severity") == "fail" for f in report.flags):
            # Nothing measurable was captured; do not manufacture numbers for it.
            return report
        return self.measure_frame(
            live, values=values, reference=reference, report=report
        )


def summarise(report: RunReport) -> str:
    """One-screen summary for a console or a CI log."""
    counts = {v.value: 0 for v in Verdict}
    for r in report.results:
        counts[r.verdict.value] += 1
    parts = [
        f"{report.screen}" + (f" [{report.theme}]" if report.theme else ""),
        f"verdict={report.verdict.value}",
        " ".join(f"{k}={v}" for k, v in counts.items()),
    ]
    if report.residual_score is not None:
        parts.append(f"ssim={report.residual_score:.4f}")
    if report.residual_findings:
        parts.append(f"residual_regions={len(report.residual_findings)}")
    if report.flags:
        parts.append(f"flags={len(report.flags)}")
    return "  ".join(parts)

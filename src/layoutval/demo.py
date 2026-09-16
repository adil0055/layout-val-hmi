"""The whole pipeline against the built-in simulator.

Run with ``layoutval demo``.  It walks the build order from the design notes --
rig, geometry, teach-in, repeatability gate, measurement, report -- and then
injects defects with a known ground truth so the numbers can be checked against
what was actually done to the display.

It is a demonstration and a regression harness, not a claim about any real rig.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from layoutval.calibration import (
    Calibration,
    Intrinsics,
    Undistorter,
    chessboard_display_points,
    homography_from_display_pattern,
)
from layoutval.capture import capture, estimate_noise_floor, median_stack
from layoutval.pipeline import Pipeline, PipelineOptions, summarise
from layoutval.profile import import_design_tree
from layoutval.report import console_table, write_junit, write_report
from layoutval.linearity import apply_linearity, run_linearity
from layoutval.repeatability import apply_study, gate, study_from_measurements
from layoutval.simulator import ClusterDisplay, SimulatedRig, VirtualCamera
from layoutval.teachin import TeachInSession
from layoutval.types import AngleModel, ElementKind, Tolerance

PATTERN = (9, 6)
SQUARE_PX = 40
PATTERN_ORIGIN = (60, 20)

# The state every reference and every measurement in the demo is taken at.
NOMINAL_STATE: dict[str, Any] = {
    "TELLTALE_BATTERY_LOW": True,
    "TELLTALE_OIL_PRESSURE": True,
    "TELLTALE_ABS": True,
    "FUEL_LEVEL": 0.6,
    "SPEED": 120.0,
}
VALUES = {"FUEL_LEVEL": 0.6, "SPEED": 120.0}

# Defects injected into the live capture, with ground truth kept alongside so
# the demo can check that the pipeline reports what was actually done.
# Offsets sit on the simulator's supersample grid so the injected value is the
# value actually rendered; otherwise the ground-truth check would be measuring
# the renderer's rounding rather than the pipeline's accuracy.
DEFECTS = {
    "TELLTALE_ABS": ("offset", (3.0, 1.25)),
    "TELLTALE_BATTERY_LOW": ("wrong_content", None),
    "NEEDLE_SPEED": ("angle", 4.0),
    "FUEL_BAR": ("offset", (2.25, 0.0)),
}


def _h(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))


def run_demo(out_dir: Path, *, inject: bool = True) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    display = ClusterDisplay()
    camera = VirtualCamera(display_size=display.size, drift_per_frame_px=0.0)
    rig = SimulatedRig(display, camera)

    # -- stage 2: intrinsics ------------------------------------------------
    _h("stage 2  undistort")
    # A real programme solves K and the distortion coefficients from a physical
    # checkerboard held at many poses.  The simulator knows its own optics, so
    # the demo uses them directly and the calibration step is exercised by
    # tests/test_calibration.py instead.
    intrinsics = Intrinsics(
        K=camera.K, dist=camera.dist, image_size=camera.sensor_size, rms=0.0
    )
    undistort = Undistorter(intrinsics)
    print(f"intrinsics: f={intrinsics.K[0, 0]:.0f} px  k1={intrinsics.dist[0]:+.3f}")

    # -- stage 3: rectification --------------------------------------------
    _h("stage 3  rectify into display space")
    rig.show("checkerboard")
    pattern_frame = undistort(median_stack(capture(rig, n=9)))
    geometry = homography_from_display_pattern(
        pattern_frame,
        PATTERN,
        chessboard_display_points(PATTERN, SQUARE_PX, PATTERN_ORIGIN),
        display_size=display.size,
    )
    rig.show("main")
    calibration = Calibration(
        intrinsics=intrinsics,
        geometry=geometry,
        rig={"source": "simulator", "note": "synthetic rig, not a real bench"},
        drift_alarm_px=2.0,
    )
    calibration.save(out_dir / "calibration.json")
    ratio = geometry.sampling_ratio()
    print(f"method={geometry.method}  fit residual={geometry.residual_px:.4f} camera px")
    print(f"sampling ratio={ratio:.2f} camera px per display px", end="")
    print("  (>=2 wanted; 3 is comfortable)" if ratio >= 2 else "  -- under-sampled")

    def capture_display() -> np.ndarray:
        return geometry.rectify(undistort(median_stack(capture(rig, n=11))))

    # -- noise floor, measured rather than guessed --------------------------
    for k, v in NOMINAL_STATE.items():
        display.state[k] = v
    noise_floor = estimate_noise_floor([rig.read() for _ in range(9)])
    print(f"measured camera noise floor: {noise_floor:.2f} grey levels")

    # -- element inventory: design export + differential teach-in -----------
    _h("inventory  design export (source A) + teach-in (source B)")
    export = json.loads((Path(__file__).resolve().parents[2] / "examples" / "design_export.json").read_text())
    profile = import_design_tree(
        export,
        screen="main",
        display_size=display.size,
        theme="day",
        defaults=Tolerance(tol_warn=1.5, tol_fail=2.5, identity_min=0.80, spec_tolerance_px=1.0),
    )
    print(f"design export: {len(profile)} elements, traceable to the design artefact")

    session = TeachInSession(
        capture_display=capture_display,
        driver=rig,
        settle_s=0.0,
        noise_floor=max(6.0, noise_floor * 4),
    )
    taught: dict[str, np.ndarray] = {}
    anchors: dict[str, tuple[float, float]] = {}
    for signal in ("TELLTALE_BATTERY_LOW", "TELLTALE_OIL_PRESSURE", "TELLTALE_ABS"):
        spec, template = session.teach(signal, kind=ElementKind.TELLTALE)
        taught[signal] = template
        # The design box is the layout box; the taught template is cut to the
        # ink inside it.  Record which, or the padding becomes a defect.
        anchors[signal] = (spec.bbox[0], spec.bbox[1])
        design = profile[signal]
        dx = spec.bbox[0] - design.bbox[0]
        dy = spec.bbox[1] - design.bbox[1]
        print(
            f"  {signal:<24} taught bbox {tuple(round(v) for v in spec.bbox)}  "
            f"vs design {tuple(round(v) for v in design.bbox)}  "
            f"delta ({dx:+.0f},{dy:+.0f}) px"
        )
    for k, v in NOMINAL_STATE.items():
        display.state[k] = v

    # The fuel bar is supposed to move: learn expected_position(signal_value)
    # rather than pinning it to a constant.
    fuel_spec, fuel_template = session.teach_moving(
        "FUEL_LEVEL", [0.0, 0.25, 0.5, 0.75, 1.0], element_id="FUEL_BAR"
    )
    taught["FUEL_BAR"] = fuel_template
    # No anchor entry: teach_moving fits the travel model from matches of this
    # very template, so the template's expected top-left *is* the model's
    # prediction and the intra-element offset is zero by construction.
    profile["FUEL_BAR"].position = fuel_spec.position
    profile["FUEL_BAR"].signal = "FUEL_LEVEL"
    profile["FUEL_BAR"].bbox = fuel_spec.bbox
    profile["FUEL_BAR"].notes = fuel_spec.notes
    # Its expected geometry no longer comes from the design export, so the
    # element must stop claiming that it does.
    profile["FUEL_BAR"].source = "teachin"
    print(
        f"  FUEL_BAR travel model: origin={tuple(round(v,1) for v in fuel_spec.position.origin)} "
        f"direction={tuple(round(v,1) for v in fuel_spec.position.direction)} px over "
        f"{fuel_spec.position.value_min}..{fuel_spec.position.value_max}"
    )
    for k, v in NOMINAL_STATE.items():
        display.state[k] = v

    # The needle is a pivot-and-angle element, not a translation one.
    needle = profile["NEEDLE_SPEED"]
    needle.kind = ElementKind.NEEDLE
    needle.pivot = display.GAUGE_PIVOT
    needle.signal = "SPEED"
    needle.angle = AngleModel(
        kind="linear", angle_at_min=210.0, angle_at_max=-30.0, value_min=0.0, value_max=240.0
    )
    # Hue-and-saturation mask: the gauge plate is desaturated and sits inside the
    # same box, so a luma threshold would fold the plate into the needle mask.
    needle.mask = {"hsv_range": [[0, 120, 80], [10, 255, 255]], "close_kernel": 3}
    needle.tolerance = Tolerance(
        tol_warn=2.0, tol_fail=3.5, angle_warn_deg=1.0, angle_fail_deg=2.0,
        spec_tolerance_px=1.5,
    )
    needle.search_margin_px = 12.0
    # No `occludes` assertion here on purpose: the needle's content changes with
    # speed, and the z-order check compares the overlap region against the
    # reference, so it only means something between elements whose content is
    # fixed.  See tests/test_residual.py for a case where it does apply.

    profile.reference_path = "reference.png"
    profile.reference_values = dict(VALUES)
    reference = capture_display()
    cv2.imwrite(str(out_dir / "reference.png"), reference)
    profile.save(out_dir / "layout_profile.yaml", templates=taught, anchors=anchors)
    print(f"profile: {len(profile)} elements -> {out_dir / 'layout_profile.yaml'}")

    # -- week 2's gate ------------------------------------------------------
    _h("gate  repeatability study")
    pipeline = Pipeline(
        calibration,
        profile,
        options=PipelineOptions(settle=False, frames_per_measurement=11),
    )
    samples: dict[str, list] = {s.id: [] for s in profile}
    n_study = 12  # a real study is ~200 frames spread over ~30 minutes
    for _ in range(n_study):
        report = pipeline.measure_frame(capture_display(), values=VALUES, reference=reference)
        for r in report.results:
            if r.element_id in samples:
                samples[r.element_id].append(r.measurement)
    study = study_from_measurements("main", samples, metadata={"rig": "simulator"})
    study.save(out_dir / "repeatability.json")
    for eid, noise in sorted(study.elements.items()):
        pick = noise.better_estimator()
        print(
            f"  {eid:<24} sigma={noise.sigma:.3f} px  3sigma={noise.resolution:.3f} px  "
            f"bias=({noise.bias_x:+.2f},{noise.bias_y:+.2f})"
            + (f"  estimator->{pick}" if pick else "")
        )
    apply_study(profile, study)

    # A static screen cannot reveal sub-pixel bias, so characterise it too.
    linearity = run_linearity(profile, reference, values=VALUES)
    linearity.save(out_dir / "linearity.json")
    apply_linearity(profile, linearity)
    print("  sub-pixel linearity (what a static-screen study cannot see):")
    for eid, lin in sorted(linearity.elements.items()):
        print(
            f"    {eid:<24} rms={lin.rms_error_px:.3f} px  max={lin.max_error_px:.3f} px"
            f"   (sigma was {study.elements[eid].sigma:.3f} px)"
        )

    result = gate(profile, study)
    print(result.report())
    print(
        "  (n=%d here for speed; a study you can put in front of a reviewer needs ~200\n"
        "   frames over ~30 minutes so warm-up drift is in the data)" % n_study
    )
    profile.save(out_dir / "layout_profile.yaml", templates=taught, anchors=anchors)

    # -- the run ------------------------------------------------------------
    _h("run  measure, score, report")
    if inject:
        for eid, (mode, arg) in DEFECTS.items():
            if mode == "offset":
                display.offsets[eid] = arg
            elif mode == "wrong_content":
                display.swapped.add(eid)
            elif mode == "angle":
                display.needle_angle_offset_deg = arg
        display.artefacts.append((520, 150, 46, 22))  # an element nobody taught
        print("injected:")
        for eid, (mode, arg) in DEFECTS.items():
            print(f"  {eid:<24} {mode}{'' if arg is None else f' {arg}'}")
        print(f"  {'(unmodelled artefact)':<24} 46x22 px at (520,150)")

    live = capture_display()
    report = pipeline.measure_frame(live, values=VALUES, reference=reference)
    print()
    print(summarise(report))
    print(console_table([report]))
    for flag in report.flags:
        print(f"  flag: {flag}")
    for finding in report.residual_findings:
        print(
            f"  residual: {finding.bbox} dissimilarity={finding.mean_dissimilarity:.2f} "
            f"overlaps={finding.overlaps or 'nothing known'}"
        )

    written = write_report(
        report, out_dir, profile=profile, live_display=live, reference=reference, values=VALUES
    )
    write_junit([report], out_dir / "junit.xml")
    print("\nwrote:")
    for k, v in {**written, "junit": out_dir / "junit.xml"}.items():
        print(f"  {k}: {v}")

    # -- did it report what was actually done? ------------------------------
    if inject:
        _h("ground truth check")
        ok = True
        for eid, (mode, arg) in DEFECTS.items():
            r = next((x for x in report.results if x.element_id == eid), None)
            if r is None:
                print(f"  {eid}: MISSING from report")
                ok = False
                continue
            m = r.measurement
            if mode == "offset":
                err = np.hypot((m.dx or 0) - arg[0], (m.dy or 0) - arg[1])
                print(
                    f"  {eid:<24} injected ({arg[0]:+.2f},{arg[1]:+.2f}) px  "
                    f"measured ({m.dx:+.2f},{m.dy:+.2f}) px  error {err:.3f} px  "
                    f"-> {r.verdict.value}/{r.reason}"
                )
                ok &= err < 0.5
            elif mode == "wrong_content":
                print(
                    f"  {eid:<24} injected wrong glyph  zncc={m.zncc:.3f}  "
                    f"-> {r.verdict.value}/{r.reason}"
                )
                ok &= r.reason == "wrong_content"
            elif mode == "angle":
                print(
                    f"  {eid:<24} injected {arg:+.2f} deg  measured "
                    f"{m.d_angle_deg:+.2f} deg  -> {r.verdict.value}/{r.reason}"
                )
                ok &= abs((m.d_angle_deg or 0) - arg) < 0.6
        print("\nground truth check: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    return 0

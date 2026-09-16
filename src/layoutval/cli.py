"""Command line entry points.

    layoutval calibrate-intrinsics  ...   stage 2, once per camera+lens
    layoutval calibrate-geometry    ...   stage 3, once per rig build
    layoutval import-design         ...   build a profile from a design export
    layoutval run                   ...   validate captured frames
    layoutval repeatability         ...   the week-2 study
    layoutval gate                  ...   is the rig good enough for these limits
    layoutval demo                  ...   the whole thing against the simulator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from layoutval.calibration import (
    Calibration,
    Intrinsics,
    Undistorter,
    calibrate_intrinsics,
    chessboard_display_points,
    homography_from_display_edges,
    homography_from_display_pattern,
)
from layoutval.capture import median_stack
from layoutval.linearity import LinearityStudy, apply_linearity, best_estimator, run_linearity
from layoutval.pipeline import Pipeline, PipelineOptions, summarise
from layoutval.profile import LayoutProfile, import_design_tree
from layoutval.report import console_table, write_junit, write_report
from layoutval.repeatability import (
    RepeatabilityStudy,
    apply_study,
    gate,
    study_from_measurements,
)
from layoutval.types import RunReport, Verdict


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image {path}")
    return img


def _frames(paths: list[str]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(_read(q) for q in sorted(path.glob("*.png")))
        else:
            out.append(_read(path))
    if not out:
        raise SystemExit("no frames found")
    return out


# --------------------------------------------------------------------------


def cmd_calibrate_intrinsics(args: argparse.Namespace) -> int:
    images = _frames(args.frames)
    intr = calibrate_intrinsics(
        images, tuple(args.pattern), args.square_size, min_views=args.min_views
    )
    print(f"reprojection error: {intr.rms:.4f} px over {len(images)} views")
    if intr.rms > args.max_rms:
        print(
            f"FAIL: reprojection error exceeds {args.max_rms} px. Re-shoot the "
            "calibration with more pose variety and check focus and exposure lock.",
            file=sys.stderr,
        )
        return 1
    Path(args.out).write_text(json.dumps(intr.to_dict(), indent=2))
    print(f"wrote {args.out}")
    return 0


def cmd_calibrate_geometry(args: argparse.Namespace) -> int:
    frame = median_stack(_frames(args.frames))
    intr = None
    if args.intrinsics:
        intr = Intrinsics.from_dict(json.loads(Path(args.intrinsics).read_text()))
        frame = Undistorter(intr)(frame)

    display_size = tuple(args.display_size)
    if args.method == "pattern":
        pattern = tuple(args.pattern)
        pts = chessboard_display_points(pattern, args.square_px, tuple(args.origin))
        geom = homography_from_display_pattern(
            frame, pattern, pts, display_size=display_size
        )
    elif args.method == "edges":
        geom = homography_from_display_edges(frame, display_size=display_size)
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"unknown method {args.method}")

    ratio = geom.sampling_ratio()
    print(f"method={geom.method} residual={geom.residual_px:.4f} px")
    print(f"sampling ratio: {ratio:.2f} camera px per display px")
    if ratio < 2.0:
        print(
            "  NOTE: below 2 camera px per display px you cannot reliably resolve a "
            "1-display-pixel shift. The honest floor here is about +/-1 display px; "
            "say so before promising anyone a tolerance.",
            file=sys.stderr,
        )
    Calibration(
        intrinsics=intr,
        geometry=geom,
        rig={"note": args.note} if args.note else {},
        drift_alarm_px=args.drift_alarm_px,
    ).save(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_import_design(args: argparse.Namespace) -> int:
    tree = json.loads(Path(args.export).read_text())
    field_map = json.loads(Path(args.field_map).read_text()) if args.field_map else None
    profile = import_design_tree(
        tree,
        screen=args.screen,
        display_size=tuple(args.display_size),
        theme=args.theme,
        field_map=field_map,
    )
    if args.reference:
        profile.reference_path = args.reference
    profile.save(args.out)
    print(f"imported {len(profile)} elements -> {args.out}")
    for problem in profile.validate():
        print(f"  warning: {problem}", file=sys.stderr)
    return 0


class _ListSource:
    """Cycles a fixed list of frames.  Used for offline replay."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0

    def read(self) -> np.ndarray:
        f = self._frames[self._i % len(self._frames)]
        self._i += 1
        return f


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    options = PipelineOptions(
        frames_per_measurement=args.stack,
        settle=False,  # offline replay: the frames are already captured
        run_residual=not args.no_residual,
    )
    return Pipeline.from_files(args.calibration, args.profile, options=options)


def cmd_run(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    frames = _frames(args.frames)
    pipeline.options.frames_per_measurement = min(
        pipeline.options.frames_per_measurement, len(frames)
    )
    values = json.loads(Path(args.values).read_text()) if args.values else None

    # Rectify once and measure that frame, rather than capturing twice: the
    # overlay must be drawn on the same pixels the verdict was formed from.
    report = RunReport(screen=pipeline.profile.screen, theme=pipeline.profile.theme)
    live = pipeline.acquire(_ListSource(frames), report)
    report = pipeline.measure_frame(live, values=values, report=report)

    print(summarise(report))
    print(console_table([report]))
    written = write_report(
        report,
        args.out,
        profile=pipeline.profile,
        live_display=live,
        reference=pipeline.profile.reference(),
        values=values,
    )
    for k, v in written.items():
        print(f"  {k}: {v}")
    if args.junit:
        print(f"  junit: {write_junit([report], args.junit)}")
    return 0 if report.verdict is not Verdict.FAIL else 1


def cmd_repeatability(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    frames = _frames(args.frames)
    reference = pipeline.profile.reference()
    samples: dict[str, list] = {s.id: [] for s in pipeline.profile}
    for frame in frames:
        live = pipeline.acquire(_ListSource([frame]))
        report = pipeline.measure_frame(live, reference=reference)
        for r in report.results:
            if r.element_id in samples:
                samples[r.element_id].append(r.measurement)

    study = study_from_measurements(pipeline.profile.screen, samples)
    study.save(args.out)
    print(f"wrote {args.out}  ({study.frames} frames)")
    for eid, noise in sorted(study.elements.items()):
        print(
            f"  {eid:<34} sigma={noise.sigma:.3f} px  3sigma={noise.resolution:.3f} px  "
            f"bias=({noise.bias_x:+.2f},{noise.bias_y:+.2f})"
        )
    result = gate(pipeline.profile, study)
    print(result.report())
    if args.apply:
        apply_study(pipeline.profile, study, set_tolerances=args.set_tolerances)
        pipeline.profile.save(args.profile)
        print(f"updated tolerances in {args.profile}")
    return 0 if result.passed else 1


def cmd_linearity(args: argparse.Namespace) -> int:
    profile = LayoutProfile.load(args.profile)
    reference = _read(Path(args.reference)) if args.reference else profile.reference()
    values = json.loads(Path(args.values).read_text()) if args.values else None
    study = run_linearity(profile, reference, values=values)
    study.save(args.out)
    print(f"wrote {args.out}")
    print(study.metadata["caveat"])
    for eid, lin in sorted(study.elements.items()):
        print(f"  {eid:<34} rms={lin.rms_error_px:.3f} px  max={lin.max_error_px:.3f} px")
    if args.choose_estimator:
        chosen = best_estimator(profile, reference, values=values)
        for eid, est in sorted(chosen.items()):
            profile[eid].estimator = est
            print(f"  {eid:<34} estimator -> {est}")
        if chosen:
            profile.save(args.profile)
            print(f"updated estimators in {args.profile}")
    if args.apply:
        apply_linearity(profile, study)
        profile.save(args.profile)
        print(f"recorded sub-pixel bias in {args.profile}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    profile = LayoutProfile.load(args.profile)
    study = RepeatabilityStudy.load(args.study)
    if args.linearity:
        apply_linearity(profile, LinearityStudy.load(args.linearity))
    result = gate(profile, study)
    print(result.report())
    return 0 if result.passed else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from layoutval.demo import run_demo

    return run_demo(Path(args.out), inject=not args.clean)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layoutval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate-intrinsics", help="solve K and distortion from checkerboard views")
    c.add_argument("frames", nargs="+")
    c.add_argument("--pattern", nargs=2, type=int, default=[9, 6])
    c.add_argument("--square-size", type=float, default=1.0)
    c.add_argument("--min-views", type=int, default=8)
    c.add_argument("--max-rms", type=float, default=0.3, help="gate on reprojection error")
    c.add_argument("--out", default="calibration/intrinsics.json")
    c.set_defaults(func=cmd_calibrate_intrinsics)

    c = sub.add_parser("calibrate-geometry", help="solve the display-space homography")
    c.add_argument("frames", nargs="+")
    c.add_argument("--display-size", nargs=2, type=int, required=True)
    c.add_argument("--method", choices=["pattern", "edges"], default="pattern")
    c.add_argument("--pattern", nargs=2, type=int, default=[9, 6])
    c.add_argument("--square-px", type=float, default=40.0)
    c.add_argument("--origin", nargs=2, type=float, default=[0.0, 0.0])
    c.add_argument("--intrinsics")
    c.add_argument("--drift-alarm-px", type=float, default=2.0)
    c.add_argument("--note", default="")
    c.add_argument("--out", default="calibration/rig.json")
    c.set_defaults(func=cmd_calibrate_geometry)

    c = sub.add_parser("import-design", help="build a profile from an HMI design-tool export")
    c.add_argument("export")
    c.add_argument("--screen", required=True)
    c.add_argument("--display-size", nargs=2, type=int, required=True)
    c.add_argument("--theme")
    c.add_argument("--field-map", help="JSON mapping this tool's field names onto the export's")
    c.add_argument("--reference", help="path to the rectified golden frame")
    c.add_argument("--out", default="profiles/layout_profile.yaml")
    c.set_defaults(func=cmd_import_design)

    for name, fn, helptext in (
        ("run", cmd_run, "validate captured frames against a profile"),
        ("repeatability", cmd_repeatability, "measure per-element noise on a static screen"),
    ):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("frames", nargs="+")
        c.add_argument("--calibration", required=True)
        c.add_argument("--profile", required=True)
        c.add_argument("--stack", type=int, default=15)
        c.add_argument("--no-residual", action="store_true")
        c.set_defaults(func=fn)
        if name == "run":
            c.add_argument("--values", help="JSON of signal values for moving elements")
            c.add_argument("--junit")
            c.add_argument("--out", default="out/run")
        else:
            c.add_argument("--out", default="out/repeatability.json")
            c.add_argument("--apply", action="store_true", help="write sigma back into the profile")
            c.add_argument(
                "--set-tolerances",
                action="store_true",
                help="derive tol_fail = 3*sigma + spec_tolerance",
            )

    c = sub.add_parser(
        "linearity",
        help="measure sub-pixel bias, which a static-screen study cannot see",
    )
    c.add_argument("--profile", required=True)
    c.add_argument("--reference", help="defaults to the profile's own reference")
    c.add_argument("--values")
    c.add_argument("--apply", action="store_true", help="record the bias in the profile")
    c.add_argument(
        "--choose-estimator",
        action="store_true",
        help="pick each element's estimator by measured sub-pixel accuracy",
    )
    c.add_argument("--out", default="out/linearity.json")
    c.set_defaults(func=cmd_linearity)

    c = sub.add_parser("gate", help="check tolerances against a repeatability study")
    c.add_argument("--profile", required=True)
    c.add_argument("--study", required=True)
    c.add_argument("--linearity", help="linearity study, so sub-pixel bias is in the floor")
    c.set_defaults(func=cmd_gate)

    c = sub.add_parser("demo", help="run the whole pipeline against the built-in simulator")
    c.add_argument("--out", default="out/demo")
    c.add_argument("--clean", action="store_true", help="do not inject defects")
    c.set_defaults(func=cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

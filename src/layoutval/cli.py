"""Command line entry points.

    layoutval charuco-board         ...   draw a bezel board to print
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
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from layoutval.calibration import (
    Calibration,
    CharucoSpec,
    DisplayGeometry,
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


#: Where a board export comes from, said at the point someone is missing one.
BOARD_HINT = (
    "It is written by the cluster: run it with --calibration-export FILE.\n"
    "       Find an existing one with:  find ~ -name board.json 2>/dev/null"
)


def _require(path_str: str, flag: str, hint: str = "") -> Path:
    """A missing or unreadable file argument, said plainly.

    These paths are typed at a shell, so the common failures are a typo, a
    relative path from the wrong directory, and a file that has not been
    produced yet. A traceback answers none of those; it just says the open
    failed, forty lines down.
    """
    path = Path(path_str).expanduser()
    if path.is_dir():
        raise SystemExit(f"error: {flag} {path} is a directory, not a file")
    if not path.is_file():
        here = Path.cwd()
        message = [f"error: {flag} {path} does not exist"]
        if not path.is_absolute():
            message.append(f"       (looked relative to {here})")
        if hint:
            message.append(f"       {hint}")
        raise SystemExit("\n".join(message))
    return path


def _read_json(path_str: str, flag: str, hint: str = "") -> Any:
    path = _require(path_str, flag, hint)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {flag} {path} could not be read as JSON: {exc}")


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
        intr = Intrinsics.from_dict(_read_json(args.intrinsics, "--intrinsics"))
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
    tree = _read_json(args.export, "export")
    field_map = _read_json(args.field_map, "--field-map") if args.field_map else None
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
    values = _read_json(args.values, "--values") if args.values else None

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
    profile = LayoutProfile.load(_require(args.profile, "--profile"))
    reference = (_read(_require(args.reference, "--reference")) if args.reference
                 else profile.reference())
    values = _read_json(args.values, "--values") if args.values else None
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
    profile = LayoutProfile.load(_require(args.profile, "--profile"))
    study = RepeatabilityStudy.load(_require(args.study, "--study"))
    if args.linearity:
        apply_linearity(profile, LinearityStudy.load(
            _require(args.linearity, "--linearity")))
    result = gate(profile, study)
    print(result.report())
    return 0 if result.passed else 1


def cmd_charuco_board(args: argparse.Namespace) -> int:
    """Draw the board to print and stick on the bezel."""
    try:
        spec = CharucoSpec.parse(args.spec)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    # Printed size is what makes the board usable, so work in millimetres and
    # let the DPI decide the pixel count, rather than emitting some pixel image
    # and leaving the scaling to whatever prints it.  The spec's lengths are
    # already in the units the board was designed in; treat them as mm.
    px_per_unit = args.dpi / 25.4
    width = int(round(spec.squares_x * spec.square_length * px_per_unit))
    height = int(round(spec.squares_y * spec.square_length * px_per_unit))
    image = spec.board().generateImage(
        (width, height), marginSize=int(round(args.margin_mm * px_per_unit))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise SystemExit(f"error: could not write {out}")

    board_w = spec.squares_x * spec.square_length
    board_h = spec.squares_y * spec.square_length
    print(f"wrote {out}  ({width}x{height} px at {args.dpi} dpi)")
    print(f"  {spec.squares_x}x{spec.squares_y} squares of "
          f"{spec.square_length:g} mm, markers {spec.marker_length:g} mm, "
          f"{spec.dictionary}")
    print(f"  prints at {board_w:g} x {board_h:g} mm "
          f"plus a {args.margin_mm:g} mm quiet margin")
    print()
    print("  Print it at 100% -- 'fit to page' rescales it and the printed size")
    print("  is what the board's own units mean. Check one square with a ruler")
    print("  before sticking it on. Matt paper or matt laminate: a glossy board")
    print("  under a cluster's own glass gives you two reflections to fight.")
    print()
    print("  Stick it on the bezel, outside the active area, flat and in the")
    print("  same plane as the screen as far as the bezel allows. Then:")
    print(f"    layoutval capture-server --charuco {args.spec} \\")
    print("        --intrinsics calibration/intrinsics.json")
    return 0


def cmd_capture_server(args: argparse.Namespace) -> int:
    from layoutval.server import (
        CaptureServer,
        CaptureSession,
        qr_terminal,
        qr_width,
    )

    pattern = tuple(args.pattern)
    display_points = None
    render = _read(_require(args.render, "--render")) if args.render else None

    # Resolve the display size from whatever actually knows it, rather than
    # leaving a default that is right for one skin and quietly wrong for the
    # other: the board export says, a render is the size it is, and only then
    # does the flag apply.
    if args.display_size:
        display_size = tuple(args.display_size)
    elif render is not None:
        display_size = (render.shape[1], render.shape[0])
        print(f"display size taken from --render: {display_size[0]}x{display_size[1]}")
    else:
        display_size = (1920, 720)

    if args.board:
        board = _read_json(args.board, "--board", BOARD_HINT)
        try:
            # corners_pixel_centre, not corners_qt: the detector puts pixel
            # column x's centre at x, and Qt puts it at x + 0.5.
            display_points = np.array(board["corners_pixel_centre"], dtype=np.float64)
            pattern = tuple(board["pattern_size"])
            if not args.display_size:
                display_size = tuple(board["canvas"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"error: {args.board} is not a board export ({exc}). It should be "
                "what the cluster's --calibration-export writes."
            )
        print(f"board: {pattern[0]}x{pattern[1]} inner corners on a "
              f"{display_size[0]}x{display_size[1]} canvas, from {args.board}")

    if render is not None and (render.shape[1], render.shape[0]) != display_size:
        raise SystemExit(
            f"error: --render is {render.shape[1]}x{render.shape[0]} but the display "
            f"is {display_size[0]}x{display_size[1]}. The render has to be the "
            "framebuffer at its own size -- from the cluster, "
            f"tools/shoot.py out.png --width {display_size[0]}"
        )

    charuco = None
    if args.charuco:
        try:
            charuco = CharucoSpec.parse(args.charuco)
        except ValueError as exc:
            raise SystemExit(f"error: --charuco: {exc}") from exc
        if not (args.intrinsics or args.calibration):
            # Refusing at startup rather than at the first capture: this route
            # cannot produce a defensible number without undistortion, and
            # finding that out after carrying a phone to the bench is worse than
            # finding it out here.
            raise SystemExit(
                "error: --charuco needs camera intrinsics, so pass --intrinsics "
                "(or a --calibration that carries them).\n"
                "       The bezel markers are photographed away from the screen's "
                "part of the frame, so lens distortion does not cancel the way it "
                "nearly does for a board on the screen.\n"
                "       Measured (benchmarks/bezel_anchor.py): undistorted this "
                "route holds 0.10 px on any lens; with the distortion left in it "
                "runs 0.6 px to 5.7 px depending on\n"
                "       the lens, and elements start dropping out of the match "
                "altogether. It is lens-dependent, so one good-looking frame "
                "tells you nothing about the next camera.\n"
                "       Solve them once for this camera and lens:\n"
                "         layoutval calibrate-intrinsics shots/*.jpg "
                "--out calibration/intrinsics.json"
            )

    profile = (LayoutProfile.load(_require(args.profile, "--profile"))
               if args.profile else None)
    calibration = (Calibration.load(_require(args.calibration, "--calibration"))
                   if args.calibration else None)
    if calibration is None and args.intrinsics:
        calibration = Calibration(
            intrinsics=Intrinsics.from_dict(_read_json(args.intrinsics, "--intrinsics")),
            geometry=DisplayGeometry(H=np.eye(3), display_size=tuple(args.display_size)),
        )

    session = CaptureSession(
        Path(args.out),
        profile=profile,
        calibration=calibration,
        pattern_size=pattern,
        square_px=args.square_px,
        pattern_origin=tuple(args.origin),
        display_points=display_points,
        display_size=display_size,
        fixed_camera=args.fixed_camera,
        drift_alarm_px=args.drift_alarm_px,
        values=_read_json(args.values, "--values") if args.values else None,
        auto_profile=not args.no_auto_profile,
        render=render,
        charuco=charuco,
    )
    server = CaptureServer(session, args.host, args.port, quiet=args.quiet)

    print()
    print("  Scan this with the phone's camera, on the same network:")
    print()
    # Skipped when it would wrap, or when this is going into a log rather than
    # to a person: a QR broken across lines is worse than no QR, because it
    # looks like something that ought to work.
    show_qr = not args.no_qr and sys.stdout.isatty()
    if show_qr:
        try:
            width = qr_width(server.url)
            columns = shutil.get_terminal_size((80, 24)).columns
            if width <= columns:
                print(qr_terminal(server.url))
                print()
            else:
                print(f"  (terminal is {columns} columns; the code needs {width})")
                print()
        except cv2.error as exc:
            print(f"  (could not draw the code: {exc})")
            print()
    print(f"      {server.url}")
    print()
    print(f"  Typing it in is fine too — the token is {server.token} and it is not")
    print("  case sensitive, with no letter O, letter l or letter i in it.")
    print()
    if profile is None and not args.no_auto_profile:
        print("  No --profile, so the elements will be found in the reference frame.")
        print("  That measures against the reference, not against the design, and")
        print("  treats everything as fixed -- keep the cluster in one state.")
        print()
    if args.render:
        print("  1  Calibrate   the screen under test -- no chessboard needed,")
        print("                 it matches the framebuffer you supplied")
    else:
        print("  1  Calibrate   cluster showing its chessboard, filling the frame")
    print("  2  Reference   cluster showing the screen under test, correct")
    print("  3  Validate    the same screen, with whatever you are testing")
    print()
    if not args.fixed_camera:
        print("  Hand-held: each frame's pose is re-solved against the reference, so")
        print("  a fault that moved every element together would be absorbed and not")
        print("  reported. Clamp the phone and pass --fixed-camera to close that gap.")
        print()
    print(f"  Captures and reports go to {Path(args.out).resolve()}")
    print("  This listens on the local network and accepts uploads. It is a bench")
    print("  tool: stop it when you are done. Ctrl-C to stop.")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


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

    c = sub.add_parser(
        "charuco-board",
        help="draw a ChArUco board to print and stick on the bezel",
    )
    c.add_argument("spec", help="COLSxROWS[:SQUARE[:MARKER[:DICT]]], the same "
                                "string capture-server's --charuco takes "
                                "(e.g. 16x3:30:22). Lengths are millimetres")
    c.add_argument("--out", default="calibration/charuco.png")
    c.add_argument("--dpi", type=float, default=600.0,
                   help="print resolution; the spec's lengths set the physical "
                        "size and this sets the pixels (default 600)")
    c.add_argument("--margin-mm", type=float, default=10.0,
                   help="white quiet margin around the board, which the marker "
                        "detector needs to find the outer squares (default 10)")
    c.set_defaults(func=cmd_charuco_board)

    c = sub.add_parser(
        "capture-server",
        help="capture from a phone on the same network, measure here",
    )
    c.add_argument("--profile", help="layout profile; without one, only calibrate and reference work")
    c.add_argument("--calibration", help="start from a stored rig calibration")
    c.add_argument("--intrinsics", help="camera intrinsics, if you have them")
    c.add_argument("--render",
                   help="the framebuffer as a display-space PNG (what the cluster "
                        "is drawing). With this, Calibrate matches the screen's own "
                        "content and no chessboard is needed at all")
    c.add_argument("--board",
                   help="the cluster's own board export (its --calibration-export). "
                        "Preferred: it carries the exact corners, so nothing has to "
                        "be guessed or converted")
    c.add_argument("--charuco", metavar="SPEC",
                   help="a ChArUco board fixed to the bezel, as COLSxROWS[:SQUARE"
                        "[:MARKER[:DICT]]] (e.g. 16x3:30:22). For a cluster that "
                        "cannot be asked to draw anything: the first Calibrate "
                        "binds the board to the active area and needs the "
                        "calibration screen too, every one after needs only the "
                        "markers. Requires --intrinsics")
    c.add_argument("--display-size", nargs=2, type=int, default=None,
                   help="the framebuffer's size; taken from --board or --render "
                        "when not given")
    c.add_argument("--pattern", nargs=2, type=int, default=[9, 6],
                   help="inner corners of the chessboard the cluster draws; "
                        "ignored when --board is given")
    c.add_argument("--square-px", type=float, default=100.0)
    c.add_argument("--origin", nargs=2, type=float, default=[0.0, 0.0],
                   help="display coordinate of the board's first inner corner region")
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--out", default="out/captures")
    c.add_argument("--values", help="JSON of signal values for moving elements")
    c.add_argument("--fixed-camera", action="store_true",
                   help="the camera is mounted: check its pose but do not re-solve it")
    c.add_argument("--drift-alarm-px", type=float, default=2.0)
    c.add_argument("--quiet", action="store_true")
    c.add_argument("--no-qr", action="store_true",
                   help="do not draw the scannable code")
    c.add_argument("--no-auto-profile", action="store_true",
                   help="without --profile, do not take an inventory from the "
                        "reference frame; refuse to validate instead")
    c.set_defaults(func=cmd_capture_server)

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

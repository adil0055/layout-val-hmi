"""Stage 6 -- structured records and annotated overlays.

The JSON is what CI consumes and what traceability hangs off; the overlay is
what a human looks at when the JSON says FAIL.  Both carry the expectation
``source`` per element, because that is what decides what a green result means.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from layoutval.profile import LayoutProfile
from layoutval.residual import dissimilarity_heatmap
from layoutval.types import ElementResult, RunReport, Verdict, resolve_value

# BGR, chosen to stay distinguishable on the dark backgrounds clusters use and
# to survive the greyscale printout somebody will inevitably attach to a defect.
COLOUR = {
    Verdict.PASS: (110, 180, 90),
    Verdict.REVIEW: (60, 175, 220),
    Verdict.FAIL: (70, 70, 225),
}
GHOST = (170, 160, 150)


def _dashed_rect(
    img: np.ndarray,
    rect: tuple[float, float, float, float],
    colour: tuple[int, int, int],
    *,
    dash: int = 6,
    gap: int = 5,
    thickness: int = 1,
) -> None:
    x, y, w, h = (int(round(v)) for v in rect)
    for x0 in range(x, x + w, dash + gap):
        x1 = min(x0 + dash, x + w)
        cv2.line(img, (x0, y), (x1, y), colour, thickness)
        cv2.line(img, (x0, y + h), (x1, y + h), colour, thickness)
    for y0 in range(y, y + h, dash + gap):
        y1 = min(y0 + dash, y + h)
        cv2.line(img, (x, y0), (x, y1), colour, thickness)
        cv2.line(img, (x + w, y0), (x + w, y1), colour, thickness)


def annotate(
    live_display: np.ndarray,
    report: RunReport,
    profile: LayoutProfile,
    *,
    values: dict[str, float] | None = None,
    label_failures_only: bool = False,
    scale: float = 1.0,
) -> np.ndarray:
    """Draw expected-versus-observed for every element onto the rectified frame.

    Expected geometry is dashed and grey, observed is solid and coloured by
    verdict, and the offset between the two centroids is drawn as an arrow.  A
    reviewer should be able to tell "correct but misplaced" from "wrong content"
    without reading the JSON.
    """
    canvas = live_display.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    by_id = {s.id: s for s in profile}
    for index, result in enumerate(report.results):
        spec = by_id.get(result.element_id)
        if spec is None:
            continue  # derived checks (z-order) have no box of their own
        m = result.measurement
        colour = COLOUR[result.verdict]
        value = resolve_value(spec, values)
        ex, ey, ew, eh = spec.expected_bbox(value)

        _dashed_rect(canvas, (ex, ey, ew, eh), GHOST)
        if m.dx is not None and m.dy is not None:
            cv2.rectangle(
                canvas,
                (int(round(ex + m.dx)), int(round(ey + m.dy))),
                (int(round(ex + m.dx + ew)), int(round(ey + m.dy + eh))),
                colour,
                1,
            )
        if m.expected_centre and m.observed_centre:
            p = tuple(int(round(v)) for v in m.expected_centre)
            q = tuple(int(round(v)) for v in m.observed_centre)
            cv2.circle(canvas, p, 2, GHOST, -1)
            cv2.circle(canvas, q, 2, colour, -1)
            if max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2:
                cv2.arrowedLine(canvas, p, q, colour, 1, tipLength=0.35)

        if label_failures_only and result.verdict is Verdict.PASS:
            continue
        delta = m.abs_delta
        label = spec.id if delta is None else f"{spec.id} {delta:.2f}px"
        if result.reason:
            label += f" {result.reason}"
        # Alternate above and below: cluster telltales sit shoulder to shoulder
        # and a single row of labels overwrites itself.
        above = index % 2 == 0
        ly = max(10, int(round(ey)) - 5) if above else min(
            canvas.shape[0] - 4, int(round(ey + eh)) + 12
        )
        cv2.putText(
            canvas, label, (int(round(ex)), ly), cv2.FONT_HERSHEY_SIMPLEX,
            0.34 * scale, colour, 1, cv2.LINE_AA,
        )

    for finding in report.residual_findings:
        x, y, w, h = finding.bbox
        _dashed_rect(canvas, (x, y, w, h), COLOUR[Verdict.REVIEW], dash=3, gap=4)
        # Below the box, so residual labels do not land on the element labels.
        cv2.putText(
            canvas,
            f"residual {finding.mean_dissimilarity:.2f}",
            (x, min(canvas.shape[0] - 4, y + h + 11)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32 * scale,
            COLOUR[Verdict.REVIEW],
            1,
            cv2.LINE_AA,
        )

    banner = f"{report.screen} {report.verdict.value}"
    if report.theme:
        banner += f" [{report.theme}]"
    cv2.putText(
        canvas, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
        COLOUR[report.verdict], 1, cv2.LINE_AA,
    )
    return canvas


def write_report(
    report: RunReport,
    out_dir: Path | str,
    *,
    profile: LayoutProfile | None = None,
    live_display: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    values: dict[str, float] | None = None,
) -> dict[str, Path]:
    """Write ``report.json`` plus, when imagery is supplied, the overlays.

    The heatmap is written whenever the residual check produced findings: the
    scalar SSIM is not something a reviewer can triage with, and a review flag
    without the picture attached gets closed unread.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    written["json"] = json_path

    if live_display is not None and profile is not None:
        overlay = annotate(live_display, report, profile, values=values)
        p = out_dir / "overlay.png"
        cv2.imwrite(str(p), overlay)
        written["overlay"] = p

        p = out_dir / "capture.png"
        cv2.imwrite(str(p), live_display)
        written["capture"] = p

    if live_display is not None and reference is not None and report.residual_findings:
        p = out_dir / "residual_heatmap.png"
        cv2.imwrite(str(p), dissimilarity_heatmap(reference, live_display))
        written["heatmap"] = p

    return written


def write_junit(
    reports: Sequence[RunReport], path: Path | str, *, suite_name: str = "layout"
) -> Path:
    """JUnit XML for CI.

    REVIEW verdicts become skipped rather than failed: a review flag that turns
    the build red teaches everyone to ignore review flags.
    """
    suites = ET.Element("testsuites", name=suite_name)
    for report in reports:
        name = report.screen + (f".{report.theme}" if report.theme else "")
        suite = ET.SubElement(suites, "testsuite", name=name)
        failures = errors = skipped = 0
        for result in report.results:
            case = ET.SubElement(
                suite, "testcase", classname=name, name=result.element_id
            )
            if result.verdict is Verdict.FAIL:
                failures += 1
                ET.SubElement(
                    case, "failure", message=result.reason or "fail"
                ).text = _detail(result)
            elif result.verdict is Verdict.REVIEW:
                skipped += 1
                ET.SubElement(
                    case, "skipped", message=result.reason or "review"
                ).text = _detail(result)
        for flag in report.flags:
            case = ET.SubElement(
                suite, "testcase", classname=name, name=f"rig.{flag['flag']}"
            )
            if flag.get("severity") == "fail":
                failures += 1
                ET.SubElement(case, "failure", message=flag["flag"]).text = json.dumps(flag)
            else:
                skipped += 1
                ET.SubElement(case, "skipped", message=flag["flag"]).text = json.dumps(flag)
        suite.set("tests", str(len(report.results) + len(report.flags)))
        suite.set("failures", str(failures))
        suite.set("errors", str(errors))
        suite.set("skipped", str(skipped))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _detail(result: ElementResult) -> str:
    from layoutval.verdict import explain

    return explain(result)


def console_table(reports: Iterable[RunReport]) -> str:
    """Fixed-width per-element table for a terminal or a CI log."""
    rows = [
        f"{'element':<34}{'verdict':<9}{'dx':>8}{'dy':>8}{'|d|':>8}{'zncc':>8}  reason"
    ]
    rows.append("-" * len(rows[0]))
    for report in reports:
        for r in report.results:
            m = r.measurement
            rows.append(
                f"{r.element_id[:33]:<34}{r.verdict.value:<9}"
                f"{_fmt(m.dx):>8}{_fmt(m.dy):>8}{_fmt(m.abs_delta):>8}"
                f"{_fmt(m.zncc, 3):>8}  {r.reason or ''}"
            )
    return "\n".join(rows)


def _fmt(v: float | None, places: int = 2) -> str:
    return "-" if v is None else f"{v:.{places}f}"

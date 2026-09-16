"""Stage 5 -- turning measurements into verdicts.

Explicit, ordered, per-element, and readable by someone who is not the person
who wrote it.  The ordering is the part that matters: identity is checked before
position, so an element rendering the wrong symbol is reported as wrong content
rather than as a position failure that sends someone into layout code for a day.
"""

from __future__ import annotations

from layoutval.types import ElementResult, ElementSpec, Measurement, Tolerance, Verdict

# Reason codes.  Kept as constants so report consumers can match on them without
# depending on prose.
WRONG_CONTENT = "wrong_content"
MISSING_OR_DISPLACED = "missing_or_displaced"
POSITION = "position"
POSITION_MARGINAL = "position_marginal"
ANGLE = "angle"
ANGLE_MARGINAL = "angle_marginal"
MEASUREMENT_ERROR = "measurement_error"
ABSENT = "absent"


def verdict_for(measurement: Measurement, tol: Tolerance) -> tuple[Verdict, str | None]:
    """Apply the ordered rules to one measurement."""
    if measurement.element_absent:
        # Checked before everything else: "not drawn at all" and "drawn wrongly"
        # are different defects and go to different people.
        return Verdict.FAIL, ABSENT

    if measurement.error:
        return Verdict.FAIL, MEASUREMENT_ERROR

    if measurement.zncc is not None and measurement.zncc < tol.identity_min:
        return Verdict.FAIL, WRONG_CONTENT

    if measurement.peak_on_search_border:
        # The correlation peak sat on the edge of the search window, so the
        # element may have moved further than the window can measure.  Unknown
        # magnitude is a failure, never a clamped value.
        return Verdict.FAIL, MISSING_OR_DISPLACED

    delta = measurement.abs_delta
    if delta is None:
        return Verdict.FAIL, MEASUREMENT_ERROR
    if delta > tol.tol_fail:
        return Verdict.FAIL, POSITION

    if measurement.d_angle_deg is not None:
        a = abs(measurement.d_angle_deg)
        if a > tol.angle_fail_deg:
            return Verdict.FAIL, ANGLE
        if a > tol.angle_warn_deg:
            return Verdict.REVIEW, ANGLE_MARGINAL

    if delta > tol.tol_warn:
        return Verdict.REVIEW, POSITION_MARGINAL

    return Verdict.PASS, None


def evaluate(spec: ElementSpec, measurement: Measurement) -> ElementResult:
    """Verdict plus the record that justifies it."""
    verdict, reason = verdict_for(measurement, spec.tolerance)
    return ElementResult(
        element_id=spec.id,
        verdict=verdict,
        reason=reason,
        measurement=measurement,
        tolerance=spec.tolerance,
        source=spec.source,
    )


def explain(result: ElementResult) -> str:
    """One line a defect report can carry verbatim."""
    m, t = result.measurement, result.tolerance
    delta = m.abs_delta
    bits = [f"{result.element_id}: {result.verdict.value}"]
    if result.reason:
        bits.append(f"({result.reason})")
    if delta is not None:
        bits.append(
            f"dx={m.dx:+.2f} dy={m.dy:+.2f} |d|={delta:.2f} px "
            f"(warn {t.tol_warn:.2f} / fail {t.tol_fail:.2f})"
        )
    if m.zncc is not None:
        bits.append(f"zncc={m.zncc:.3f} (min {t.identity_min:.2f})")
    if m.d_angle_deg is not None:
        bits.append(f"dangle={m.d_angle_deg:+.2f} deg")
    if m.error:
        bits.append(f"error={m.error}")
    return " ".join(bits)

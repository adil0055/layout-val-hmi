"""The ordered verdict rules.

The ordering is the point: identity before position, so an element rendering the
wrong symbol is reported as wrong content and not as a position failure that
sends someone into layout code for a day.
"""

from __future__ import annotations

from layoutval.types import ElementSpec, Measurement, Tolerance, Verdict
from layoutval.verdict import (
    ABSENT,
    MEASUREMENT_ERROR,
    MISSING_OR_DISPLACED,
    POSITION,
    POSITION_MARGINAL,
    WRONG_CONTENT,
    evaluate,
    explain,
    verdict_for,
)

TOL = Tolerance(tol_warn=1.5, tol_fail=2.5, identity_min=0.8, angle_warn_deg=1.0, angle_fail_deg=2.0)


def m(**kw) -> Measurement:
    base = dict(element_id="e", dx=0.0, dy=0.0, zncc=0.99)
    base.update(kw)
    return Measurement(**base)


def test_pass():
    assert verdict_for(m(), TOL) == (Verdict.PASS, None)


def test_wrong_content_beats_position():
    """A big offset AND a bad ZNCC must be reported as wrong content."""
    v, reason = verdict_for(m(dx=9.0, dy=9.0, zncc=0.2), TOL)
    assert (v, reason) == (Verdict.FAIL, WRONG_CONTENT)


def test_absent_beats_everything():
    v, reason = verdict_for(m(element_absent=True, zncc=0.2, error="no lit pixels"), TOL)
    assert (v, reason) == (Verdict.FAIL, ABSENT)


def test_border_peak_is_a_failure_not_a_clamped_pass():
    v, reason = verdict_for(m(dx=0.1, dy=0.1, peak_on_search_border=True), TOL)
    assert (v, reason) == (Verdict.FAIL, MISSING_OR_DISPLACED)


def test_position_bands():
    assert verdict_for(m(dx=1.0), TOL) == (Verdict.PASS, None)
    assert verdict_for(m(dx=2.0), TOL) == (Verdict.REVIEW, POSITION_MARGINAL)
    assert verdict_for(m(dx=3.0), TOL) == (Verdict.FAIL, POSITION)


def test_angle_bands():
    assert verdict_for(m(d_angle_deg=0.5), TOL)[0] is Verdict.PASS
    assert verdict_for(m(d_angle_deg=1.5), TOL) == (Verdict.REVIEW, "angle_marginal")
    assert verdict_for(m(d_angle_deg=-3.0), TOL) == (Verdict.FAIL, "angle")


def test_missing_measurement_is_a_failure():
    assert verdict_for(m(dx=None, dy=None), TOL) == (Verdict.FAIL, MEASUREMENT_ERROR)
    assert verdict_for(m(error="boom"), TOL) == (Verdict.FAIL, MEASUREMENT_ERROR)


def test_identity_check_is_skipped_when_there_is_no_zncc():
    """A needle has no ZNCC; that must not read as zero and fail it."""
    assert verdict_for(m(zncc=None), TOL) == (Verdict.PASS, None)


def test_explain_is_readable():
    spec = ElementSpec(id="TELLTALE_ABS", tolerance=TOL)
    text = explain(evaluate(spec, m(dx=3.0, dy=0.4, zncc=0.97)))
    assert "TELLTALE_ABS" in text and "FAIL" in text and "position" in text
    assert "3.00" in text and "zncc" in text


def test_worst_verdict_aggregation():
    assert Verdict.worst([Verdict.PASS, Verdict.REVIEW]) is Verdict.REVIEW
    assert Verdict.worst([Verdict.REVIEW, Verdict.FAIL]) is Verdict.FAIL
    assert Verdict.worst([]) is Verdict.PASS

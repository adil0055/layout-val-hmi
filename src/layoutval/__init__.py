"""Camera-based layout validation for automotive HMI instrument clusters.

The design position this package implements: layout validation is a
*measurement* problem, not a detection problem.  The camera is fixed, the
cluster is fixed and the element inventory is known before the test runs, so
nothing needs to be discovered at runtime -- it needs to be measured, in
display pixels, against a design reference.

Learned models belong in the authoring loop (see :mod:`layoutval.authoring`);
the measurement loop is deterministic code only.
"""

from layoutval.types import (
    AngleModel,
    ElementKind,
    ElementResult,
    ElementSpec,
    Measurement,
    PositionModel,
    RunReport,
    Tolerance,
    Verdict,
)

__all__ = [
    "AngleModel",
    "ElementKind",
    "ElementResult",
    "ElementSpec",
    "Measurement",
    "PositionModel",
    "RunReport",
    "Tolerance",
    "Verdict",
    "__version__",
]

__version__ = "0.1.0"

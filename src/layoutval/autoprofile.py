"""An inventory taken from the reference frame, when there is no authored one.

The rest of this package assumes the element inventory is known before the test
runs -- exported from the design tool, or taught by toggling CAN signals. That
is the right way round, and it is what makes a result mean "the build matches
the design".

It is also a piece of work, and until it exists there is nothing to measure. So
this is the fallback: a cluster is bright elements on a dark background, which
segments cleanly, and each lit region can be treated as an element whether or
not anyone has named it. No authoring, no CAN, no design export.

**What this buys and what it costs.** It answers "does this frame match the
reference frame", to the same sub-pixel accuracy as an authored profile, which
is exactly the question worth asking when a fault has been deliberately
injected. It does not answer "does the build match the design": the expectation
is a photograph, so a layout error present when the reference was taken is
baked into the reference and will never be reported. It also cannot name
anything -- a defect comes back as ``auto@312,75`` rather than
``TELLTALE_ABS``, and the annotated overlay is what turns that back into an
element.

The other thing to know is that everything the cluster draws is treated as
fixed. A needle at a different speed or a bar at a different level is a moving
element to an authored profile and a failure to this one, so the cluster has to
be in the same state for the reference and for the frames measured against it.
"""

from __future__ import annotations

import cv2
import numpy as np

from layoutval.capture import to_gray
from layoutval.profile import LayoutProfile
from layoutval.types import ElementKind, ElementSpec, PositionModel, Tolerance


def segment_reference(
    reference: np.ndarray,
    *,
    threshold: int | None = None,
    min_area_px: int = 80,
    max_area_fraction: float = 0.25,
    close_kernel: int = 5,
) -> list[tuple[int, int, int, int, int]]:
    """Lit regions of a rectified frame, as ``(x, y, w, h, area)``.

    ``max_area_fraction`` drops anything covering more than that much of the
    frame: an "element" the size of the cluster measures nothing and hides
    everything inside it.

    **The threshold corrects itself, because Otsu alone is not safe here.**
    Otsu is a two-class split, and a rectified frame often has three
    populations rather than two -- black padding outside the display, the
    screen's own dark background, and the lit artwork. Measured on a frame
    rectified from hand-marked corners, with 29% of it black padding, Otsu
    landed at 42: between the padding and everything else, leaving the whole
    cluster as one blob covering 71% of the frame. Every element inside it was
    swallowed and the profile came back with one entry. At 120 the same frame
    gives 91.

    So when the segmentation produces a component covering more of the frame
    than an element ever should, that is evidence the threshold is too low, and
    it is raised until that stops being true. The same correction covers the
    unusually bright screen the old code could only warn about.
    """
    gray = to_gray(reference)
    ceiling = max_area_fraction * gray.size

    if threshold is not None:
        levels = [int(threshold)]
    else:
        # The value OpenCV chose, not one re-derived from the mask: taking the
        # smallest surviving pixel gives otsu+1, and re-thresholding at that
        # drops a grey level that Otsu kept.
        otsu_value, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        otsu = int(otsu_value)
        # Otsu first, then percentiles of the frame's own brightness. Lit
        # artwork is a small fraction of a cluster, so the high percentiles are
        # where the boundary between background and content actually sits.
        # Otsu first, then only levels ABOVE it. Sorting the whole set instead
        # starts the search below Otsu, where a lower threshold happens to
        # produce no oversized component and the loop stops there -- on a
        # frame where Otsu was right, that turned 6 correct elements into 8
        # wrong ones.
        higher = sorted({
            int(np.percentile(gray, q)) for q in (75, 85, 92, 96, 98)
        })
        levels = [otsu] + [v for v in higher if v > otsu]
        levels = [max(1, min(254, v)) for v in levels]

    kernel = np.ones((close_kernel, close_kernel), np.uint8) if close_kernel > 1 else None
    best: list[tuple[int, int, int, int, int]] = []
    for level in levels:
        _, mask = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)

        oversized = any(stats[i, cv2.CC_STAT_AREA] > ceiling for i in range(1, count))
        found = []
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (min_area_px <= area <= ceiling):
                continue
            found.append((
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
                area,
            ))
        if not oversized and found:
            # Nothing is swallowing the screen at this level, so stop here.
            # Climbing further to collect more components is a mistake: a
            # higher threshold breaks real elements into fragments, and
            # fragments do not match between reference and validate. Tried the
            # other way round first, maximising the count, and it turned a
            # passing run into "6 pass, 31 fail".
            best = found
            break
        if not best:
            best = found

    # Reading order, so the ids come out the same way twice and a report reads
    # down the screen rather than in whatever order the labeller happened to go.
    best.sort(key=lambda r: (r[1], r[0]))
    return best


def profile_from_reference(
    reference: np.ndarray,
    *,
    screen: str = "auto",
    theme: str | None = None,
    display_size: tuple[int, int] | None = None,
    defaults: Tolerance | None = None,
    max_elements: int = 80,
    **segment_kw,
) -> LayoutProfile:
    """Build a measurable inventory out of a reference frame.

    Elements are named for where they are, because there is nothing else to name
    them after, and that id is stable for a given reference.
    """
    height, width = reference.shape[:2]
    profile = LayoutProfile(
        screen=screen,
        display_size=display_size or (width, height),
        theme=theme,
        defaults=defaults or Tolerance(),
    )
    regions = segment_reference(reference, **segment_kw)
    # Biggest first when there are too many: a cap that kept the top-left corner
    # of the screen and dropped the instruments would be the wrong forty.
    if len(regions) > max_elements:
        regions = sorted(regions, key=lambda r: r[4], reverse=True)[:max_elements]
        regions.sort(key=lambda r: (r[1], r[0]))

    for x, y, w, h, area in regions:
        profile.add(ElementSpec(
            id=f"auto@{x},{y}",
            kind=ElementKind.REGION,
            bbox=(float(x), float(y), float(w), float(h)),
            position=PositionModel(origin=(float(x), float(y))),
            tolerance=profile.defaults,
            source="auto",
            notes=(
                f"found in the reference frame, {w}x{h} px, {area} lit px. "
                "Not authored: this measures against the reference capture, not "
                "against the design."
            ),
        ))
    profile.metadata["inventory"] = "auto"
    profile.metadata["note"] = (
        "Discovered by segmenting the reference frame. Answers whether a frame "
        "matches that reference, not whether the build matches the design, and "
        "treats everything as fixed -- the cluster must be in the same state "
        "for the reference and for what is measured against it."
    )
    return profile

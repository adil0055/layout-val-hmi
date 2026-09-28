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
    min_area_px: int = 40,
    min_thickness_px: int = 9,
    max_area_fraction: float = 0.25,
    close_kernel: int = 5,
) -> list[tuple[int, int, int, int, int]]:
    """Lit regions of a rectified frame, as ``(x, y, w, h, area)``.

    **Size is set by what a hand-held photograph can measure.** ``min_area_px``
    was 80, which left out a speedometer's tick marks (48-66 px) that a person
    plainly sees. Taken down to 30 they all came in, and so did elements a few
    pixels thick that a slightly shaken photo smears past recognition: under a
    12 px shake streak, 19 of 127 elements of 30-80 px failed at least once,
    against none above 200 px. What decides it is thickness against the
    smear, so a region must also be ``min_thickness_px`` across its narrower
    side: at 9, shake failures were back to the level of the old rule while
    most tick marks stay in.

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
            # Thinner than a hand's smear cannot be measured from a hand-held
            # photo (above). It also covers the hairline -- a box outline 223 px
            # long and 2 high -- whose template was under the 3 px the
            # correlator needs, and came back as a measurement error on a
            # frame compared with itself.
            if min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < min_thickness_px:
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


def structure(gray: np.ndarray, box: tuple[int, int, int, int], pad: int = 3) -> float:
    """How two-dimensional the detail in ``box`` is: 0 for a straight line, 1 for a dot.

    The smaller eigenvalue of the gradient structure tensor over the larger --
    the corner test of Shi and Tomasi. A straight line or edge looks the same
    all along its length, so nothing can say where along it an element sits:
    compared with itself, a 370 px edge on a bench photograph matched 8 px
    down its own length and came back FAIL. Measured on two bench photographs,
    lines score under 0.02, a long bar 0.07, and every tick mark, digit and
    icon 0.12 or more. The cut was 0.05 until a long bar under glare slid
    along its own length on a screen that had not changed; it is 0.10.
    """
    x, y, w, h = box
    H, W = gray.shape[:2]
    patch = gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    patch = patch.astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    a, b, c = float((gx * gx).sum()), float((gx * gy).sum()), float((gy * gy).sum())
    half, root = (a + c) / 2.0, float(np.sqrt(max(((a - c) / 2.0) ** 2 + b * b, 0.0)))
    return (half - root) / (half + root) if half + root > 0 else 0.0


def crispness(gray: np.ndarray, box: tuple[int, int, int, int], pad: int = 3) -> float:
    """How crisp a region's sharpest detail is: contrast at the scale of strokes.

    A light blur against a wider one, so neither pixel noise nor a smooth
    gradient counts, and the 98th percentile of it over the region -- its
    sharpest detail, not its average, or a speedometer's crisp ticks are
    averaged away over the soft glow round them and the whole dial goes. (It
    did, measured as a standard deviation.) Text and icons are crisp; a faint
    patch of a bar's gradient fill is not -- and those are exactly what glare
    changed the look of and the matcher slid along: on a bench photograph,
    5-8 px "wrong content" on a screen that had not changed. Measured there,
    those patches scored 24-26 and the screen's text 78-103; the dial and the
    gauges 44-60.
    """
    x, y, w, h = box
    H, W = gray.shape[:2]
    p = gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    p = p.astype(np.float32)
    detail = np.abs(cv2.GaussianBlur(p, (0, 0), 0.7) - cv2.GaussianBlur(p, (0, 0), 4.0))
    return float(np.percentile(detail, 98))


def profile_from_reference(
    reference: np.ndarray,
    *,
    screen: str = "auto",
    theme: str | None = None,
    display_size: tuple[int, int] | None = None,
    defaults: Tolerance | None = None,
    max_elements: int = 200,
    valid: np.ndarray | None = None,
    min_structure: float = 0.10,
    min_crispness: float = 0.40,
    **segment_kw,
) -> LayoutProfile:
    """Build a measurable inventory out of a reference frame.

    Elements are named for where they are, because there is nothing else to name
    them after, and that id is stable for a given reference.

    ``valid`` marks the part of the frame the camera actually saw; a region
    touching anything else is the edge of the photograph, not an element.
    Regions with less 2-D detail than ``min_structure`` (see :func:`structure`)
    are left out: their position along their own length cannot be measured.
    So are regions with less fine detail than ``min_crispness`` of the frame's
    typical element (see :func:`crispness`).
    """
    height, width = reference.shape[:2]
    profile = LayoutProfile(
        screen=screen,
        display_size=display_size or (width, height),
        theme=theme,
        defaults=defaults or Tolerance(),
    )
    regions = segment_reference(reference, **segment_kw)
    gray = to_gray(reference)
    # Nothing hugging the frame's edge: that is the display's rim, or a desktop
    # status icon on a laptop under test, not the HMI -- and the frame's
    # corners are where a hand-held pose is least certain. Status icons in the
    # top-right corner of a bench photograph read 1.6-2.7 px off under shake.
    edge = round(0.02 * min(height, width))
    regions = [r for r in regions if r[0] >= edge and r[1] >= edge
               and r[0] + r[2] <= width - edge and r[1] + r[3] <= height - edge]
    if valid is not None:
        seen = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        regions = [r for r in regions if seen[r[1]:r[1] + r[3], r[0]:r[0] + r[2]].all()]
    regions = [r for r in regions if structure(gray, r[:4]) >= min_structure]
    if regions:
        crisp = [crispness(gray, r[:4]) for r in regions]
        floor = min_crispness * float(np.median(crisp))
        regions = [r for r, c in zip(regions, crisp) if c >= floor]
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

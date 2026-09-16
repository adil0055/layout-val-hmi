"""Differential teach-in -- the workhorse source for the element inventory.

The trick is that the label is correct *by construction*.  You toggle a known
CAN signal, diff the two median-stacked captures, and whatever changed is that
signal's element.  No annotation pass, no labelling errors, and no drift between
what the authoring tool thinks an element is called and what the CAN matrix
calls it.

For elements that cannot be toggled -- a dial face, static text, a permanently
visible gauge -- drive the underlying value across its range instead and diff.
What moves is the needle or the digits; what stays is the static plate.  That
separation is useful on its own: the two need different validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np

from layoutval.capture import to_gray
from layoutval.measure import zncc_match
from layoutval.profile import LayoutProfile
from layoutval.types import ElementKind, ElementSpec, PositionModel, Tolerance


class SignalDriver(Protocol):
    """Whatever puts the cluster into a given state.

    In practice a CAN gateway, a diagnostic session or a HIL rig.  The pipeline
    does not care which, only that setting a signal and waiting is possible.
    """

    def set(self, signal: str, value: Any) -> None:  # pragma: no cover - protocol
        ...


@dataclass
class TeachInSession:
    """Capture + drive, bound together, in display space.

    ``capture_display`` must return an already rectified, median-stacked frame:
    teach-in inherits the whole accuracy budget of the geometry stage, and an
    inventory taught in camera space is invalid the moment the rig is touched.
    """

    capture_display: Callable[[], np.ndarray]
    driver: SignalDriver
    settle_s: float = 0.4
    noise_floor: float = 6.0
    """Difference threshold in 8-bit grey levels.  Measure it with
    :func:`layoutval.capture.estimate_noise_floor` on a static screen rather than
    guessing; too low and the whole screen is an element, too high and a dim
    telltale is invisible."""

    min_area_px: int = 24
    close_kernel: int = 5
    adaptive_floor: bool = True
    """Derive the difference threshold from two captures of the *same* state.

    A fixed threshold is not safe here.  Backlight PWM leaves residual banding
    that median-stacking reduces but does not remove, and it is proportional to
    brightness -- so a large bright element can differ from itself by more than a
    small telltale differs from its own off state.  Teach-in then silently
    returns the wrong region, with a plausible-looking box, correctly labelled
    with the signal you toggled.
    """

    floor_percentile: float = 99.99
    """Near the top of the same-state difference, not its median.

    What matters is the largest difference the screen shows against itself, not
    the typical one -- a handful of surviving pixels become a component once the
    closing operation joins them up.  Not the outright maximum, so a single hot
    pixel cannot raise the floor above a dim element.
    """

    floor_margin: float = 2.0
    _log: list[str] = field(default_factory=list)

    def _apply(self, signal: str, value: Any) -> np.ndarray:
        self.driver.set(signal, value)
        if self.settle_s:
            time.sleep(self.settle_s)
        return self.capture_display()

    # -- core primitive ------------------------------------------------------

    def measured_floor(self, a: np.ndarray, b: np.ndarray) -> float:
        """How much the screen differs from itself, in grey levels.

        Two captures of the same state.  Whatever comes out of this is not a
        signal, so nothing below it can be treated as one.
        """
        d = cv2.absdiff(to_gray(a), to_gray(b))
        return float(np.percentile(d, self.floor_percentile)) + self.floor_margin

    def difference_mask(
        self, off: np.ndarray, on: np.ndarray, *, floor: float | None = None
    ) -> np.ndarray:
        d = cv2.absdiff(to_gray(on), to_gray(off))
        _, mask = cv2.threshold(d, floor if floor is not None else self.noise_floor, 255, cv2.THRESH_BINARY)
        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _floor_for(self, *frames: np.ndarray) -> float:
        """Floor over every pair of same-state captures given.

        Pass at least three.  Two consecutive bursts under-report: the PWM phase
        advances per burst, so frames two apart can differ by more than frames
        one apart, and the toggled capture is two bursts away from the baseline.
        """
        if not self.adaptive_floor or len(frames) < 2:
            return self.noise_floor
        pairs = [
            self.measured_floor(frames[i], frames[j])
            for i in range(len(frames))
            for j in range(i + 1, len(frames))
        ]
        return max(self.noise_floor, max(pairs))

    def _baseline(self, first: np.ndarray, extra: int = 2) -> tuple[float, np.ndarray]:
        """Extra same-state captures, and the floor they imply."""
        frames = [first] + [self.capture_display() for _ in range(extra)]
        return self._floor_for(*frames), frames[0]

    def teach(
        self,
        signal: str,
        *,
        element_id: str | None = None,
        kind: ElementKind = ElementKind.TELLTALE,
        off_value: Any = False,
        on_value: Any = True,
        mode: str = "largest",
        tolerance: Tolerance | None = None,
    ) -> tuple[ElementSpec, np.ndarray]:
        """Toggle one signal and learn the element it controls.

        ``mode='largest'`` takes the biggest connected component, which is right
        for a single symbol.  ``mode='union'`` takes the bounding box of every
        component above ``min_area_px``, which is right for a symbol that comes
        with a text label attached -- but check the result, because a union that
        has quietly swallowed half the screen is the usual sign that
        ``noise_floor`` is too low.
        """
        off = self._apply(signal, off_value)
        floor, off = self._baseline(off) if self.adaptive_floor else (self.noise_floor, off)
        on = self._apply(signal, on_value)
        mask = self.difference_mask(off, on, floor=floor)

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        keep = [
            i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px
        ]
        if not keep:
            raise RuntimeError(
                f"{signal}: nothing changed above the measured floor "
                f"({floor:.1f} grey levels). Either the signal did not take effect, "
                "the element was already in that state, or the element is dimmer "
                "than the rig's own frame-to-frame variation -- in which case fix "
                "the rig (longer exposure, more stacked frames) rather than lowering "
                "the threshold."
            )

        if mode == "largest":
            i = max(keep, key=lambda j: stats[j, cv2.CC_STAT_AREA])
            x, y, w, h = (int(stats[i, k]) for k in range(4))
            area = float(stats[i, cv2.CC_STAT_AREA])
            cx, cy = (float(v) for v in centroids[i])
        elif mode == "union":
            xs = [int(stats[i, cv2.CC_STAT_LEFT]) for i in keep]
            ys = [int(stats[i, cv2.CC_STAT_TOP]) for i in keep]
            x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
            y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
            x, y = min(xs), min(ys)
            w, h = x1 - x, y1 - y
            area = float(sum(stats[i, cv2.CC_STAT_AREA] for i in keep))
            weights = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep], dtype=np.float64)
            cx, cy = (centroids[keep] * weights[:, None]).sum(axis=0) / weights.sum()
        else:
            raise ValueError(f"unknown teach mode {mode!r}")

        eid = element_id or signal
        spec = ElementSpec(
            id=eid,
            kind=kind,
            bbox=(float(x), float(y), float(w), float(h)),
            position=PositionModel(origin=(float(x), float(y))),
            tolerance=tolerance or Tolerance(),
            source="teachin",
            signal=signal,
            notes=(
                f"taught by toggling {signal}; {len(keep)} component(s), "
                f"{area:.0f} changed px, centroid ({cx:.1f}, {cy:.1f})"
            ),
        )
        template = on[y : y + h, x : x + w].copy()
        self._log.append(f"{eid}: bbox=({x},{y},{w},{h}) area={area:.0f} floor={floor:.1f}")
        return spec, template

    # -- elements that are supposed to move ---------------------------------

    def teach_moving(
        self,
        signal: str,
        values: Sequence[float],
        *,
        element_id: str | None = None,
        kind: ElementKind = ElementKind.REGION,
        tolerance: Tolerance | None = None,
    ) -> tuple[ElementSpec, np.ndarray]:
        """Learn an ``expected_position(signal_value)`` model by sweeping a signal.

        Progress bars, fuel bars and scrolling lists have an expected position
        that is a function of state.  Modelling them as constants means either
        testing nothing (a tolerance wide enough to cover the travel) or failing
        constantly.

        The element's extent is taken from the **signed** difference between the
        two ends of the sweep: pixels that got brighter are where it travelled
        to, pixels that got darker are where it came from.  An unsigned
        ``absdiff`` would merge the two into one blob whenever the positions
        overlap, and silently return a box spanning the whole travel -- which is
        the sort of bug that produces a plausible-looking profile and nonsense
        verdicts.

        Intermediate sweep points are then located by matching that template, and
        a straight line is fitted through every measured origin.  The residual of
        the fit is recorded: a large one means the travel is not linear and this
        model is the wrong shape.
        """
        if len(values) < 3:
            raise ValueError("need at least three sweep points to fit a travel model")

        vmin, vmax = float(min(values)), float(max(values))
        # The self-difference has to come from two captures of the *same* state,
        # so take the second one before the sweep moves on.
        frame_min = self._apply(signal, vmin)
        floor, frame_min = (
            self._baseline(frame_min) if self.adaptive_floor else (self.noise_floor, frame_min)
        )
        frames = {float(vmin): frame_min}
        for v in values:
            if float(v) != vmin:
                frames[float(v)] = self._apply(signal, v)
        frame_max = frames[vmax]

        signed = to_gray(frame_max).astype(np.int16) - to_gray(frame_min).astype(np.int16)
        box_max = self._largest_box((signed > floor).astype(np.uint8) * 255)
        box_min = self._largest_box((signed < -floor).astype(np.uint8) * 255)
        if box_max is None or box_min is None:
            raise RuntimeError(
                f"{signal}: the sweep did not move anything above the measured floor "
                f"({floor:.1f} grey levels)"
            )

        w = float(max(box_min[2], box_max[2]))
        h = float(max(box_min[3], box_max[3]))
        if min(box_min[2], box_max[2]) < 0.6 * w or min(box_min[3], box_max[3]) < 0.6 * h:
            # The two extremes still overlap, so each signed region is only the
            # part that did not coincide.  Widen the sweep or the element is not
            # a simple translator.
            self._log.append(
                f"{element_id or signal}: sweep extremes overlap "
                f"({box_min[2]}x{box_min[3]} vs {box_max[2]}x{box_max[3]}); "
                "the travel model may be wrong"
            )

        template = frame_max[
            box_max[1] : box_max[1] + int(h), box_max[0] : box_max[0] + int(w)
        ].copy()

        span_x0 = min(box_min[0], box_max[0])
        span_y0 = min(box_min[1], box_max[1])
        span_x1 = max(box_min[0] + box_min[2], box_max[0] + box_max[2])
        span_y1 = max(box_min[1] + box_min[3], box_max[1] + box_max[3])
        pad = 6
        band = (
            max(0, span_x0 - pad),
            max(0, span_y0 - pad),
            span_x1 - span_x0 + 2 * pad,
            span_y1 - span_y0 + 2 * pad,
        )

        # Every point, the two extremes included, is located by matching the same
        # template.  Mixing template-matched positions with threshold-derived
        # bounding boxes would put a sub-pixel step into the fitted model at the
        # ends of the travel, where it is least visible and most annoying.
        observed: list[tuple[float, float, float]] = []
        for v, frame in frames.items():
            area = frame[band[1] : band[1] + band[3], band[0] : band[0] + band[2]]
            if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
                continue
            (px, py), _, _ = zncc_match(area, template)
            observed.append((v, band[0] + px, band[1] + py))
        if len(observed) < 2:
            raise RuntimeError(f"{signal}: sweep produced fewer than two usable positions")

        vs = np.array([o[0] for o in observed], dtype=np.float64)
        xs = np.array([o[1] for o in observed], dtype=np.float64)
        ys = np.array([o[2] for o in observed], dtype=np.float64)
        span = (vmax - vmin) or 1.0
        A = np.column_stack([(vs - vmin) / span, np.ones_like(vs)])
        (mx, bx), *_ = np.linalg.lstsq(A, xs, rcond=None)
        (my, by), *_ = np.linalg.lstsq(A, ys, rcond=None)
        residual = float(
            np.sqrt(np.mean((A @ [mx, bx] - xs) ** 2 + (A @ [my, by] - ys) ** 2))
        )

        eid = element_id or signal
        spec = ElementSpec(
            id=eid,
            kind=kind,
            bbox=(float(bx), float(by), w, h),
            position=PositionModel(
                kind="linear",
                origin=(float(bx), float(by)),
                direction=(float(mx), float(my)),
                value_min=vmin,
                value_max=vmax,
            ),
            tolerance=tolerance or Tolerance(),
            source="teachin",
            signal=signal,
            notes=(
                f"travel model fitted over {len(observed)} sweep points; "
                f"residual {residual:.2f} px -- a large residual means the travel is "
                "not linear and this model is the wrong shape"
            ),
        )
        return spec, template

    def _largest_box(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px]
        if not keep:
            return None
        i = max(keep, key=lambda j: stats[j, cv2.CC_STAT_AREA])
        return tuple(int(stats[i, k]) for k in range(4))  # type: ignore[return-value]

    def split_static_and_moving(
        self, signal: str, values: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate the moving part of a gauge from its static plate.

        Returns ``(moving_mask, static_mask)`` over the whole display.  The two
        need different validation: the plate is a translation check, the needle
        is a pivot-and-angle check.
        """
        first = self._apply(signal, values[0])
        floor, first = (
            self._baseline(first) if self.adaptive_floor else (self.noise_floor, first)
        )
        frames = [first] + [self._apply(signal, v) for v in values[1:]]
        stack = np.stack([to_gray(f).astype(np.float32) for f in frames], axis=0)
        spread = stack.max(axis=0) - stack.min(axis=0)
        moving = (spread > floor).astype(np.uint8) * 255
        lit = (stack.mean(axis=0) > floor).astype(np.uint8) * 255
        static = cv2.bitwise_and(lit, cv2.bitwise_not(moving))
        return moving, static


def teach_profile(
    session: TeachInSession,
    signals: Sequence[str | dict[str, Any]],
    *,
    screen: str,
    display_size: tuple[int, int],
    theme: str | None = None,
    defaults: Tolerance | None = None,
) -> tuple[LayoutProfile, dict[str, np.ndarray]]:
    """Teach a whole screen.  Returns the profile and its templates.

    Failures are collected rather than raised, so one signal that did not take
    effect does not throw away the other forty elements of a teach-in run; the
    problem lands in ``profile.metadata['teach_failures']`` where it is visible
    in the stored artefact.
    """
    profile = LayoutProfile(
        screen=screen, display_size=display_size, theme=theme, defaults=defaults
    )
    templates: dict[str, np.ndarray] = {}
    failures: list[str] = []
    for entry in signals:
        opts = {"signal": entry} if isinstance(entry, str) else dict(entry)
        signal = opts.pop("signal")
        if "kind" in opts:
            opts["kind"] = ElementKind(opts["kind"])
        try:
            spec, template = session.teach(signal, tolerance=defaults, **opts)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        profile.add(spec)
        templates[spec.id] = template
    if failures:
        profile.metadata["teach_failures"] = failures
    return profile, templates

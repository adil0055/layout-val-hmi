"""CAN teach-in: find, name and learn every lamp and element a cluster draws.

The cluster is driven over CAN through a *teach plan* (see :class:`TeachPlan`):
every lamp switched on alone, then all of them at once, then each element that
shows a value driven through a few values. Each step is compared with the rest
state, in display space, so what changed is that step's by construction --
the name comes from the signal that was sent, not from anyone labelling a
picture.

Lamps are learnt by their **symbol**: the change their lighting makes, kept as
a template, with their lens colour and whether they flash. A lamp is
recognised by what it looks like, not by where it is, and then its place is
checked. On a production cluster every lamp has a slot of its own and lights
there whatever else is lit; a lamp found at the same place alone, with all
lamps lit and in the bulb check is taken to be *fixed* and is held to that
place in every state. Some displays instead show only the lit lamps, closed
up in a strip -- every lamp lit alone then sits in the same middle slot -- and
a lamp whose place changes like that is checked only in the states it was
taught in (alone, all lit, bulb check), and by its order along the strip.

Elements are named by the sweep that changes them: an element found on the
rest screen whose own box changes in most of a sweep's states is that sweep's
element ("Speed", "Gear"). A needle crossing a scale numeral changes it in one
state at most, so the numeral is not taken for the speed. Whatever no step
changes is reported as unknown, which is a finding, not a failure.

Frames here are display-space BGR images, rectified exactly as the reference
is; the camera must not move during a teach-in.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: Smallest brightening, 8-bit levels, that counts as a lamp lighting.
LIT_MIN = 20.0
#: Chroma counts this much more than brightness: a dim red lamp on grey moves
#: the colour axes far more than the value.
CHROMA_WEIGHT = 1.4
#: Smallest change, 8-bit levels, that counts as an element changing.
CHANGE_MIN = 24.0
#: A template match below this is not that lamp. Through a simulated webcam a
#: lamp's own symbol scored 0.998-1.0 and a symbol no lamp has 0.68-0.73 against
#: its nearest lamp; a real camera's focus and moire leave less, so the line sits
#: between.
MATCH_MIN = 0.82
#: An unknown symbol says which lamp it looks most like, from this score up.
LOOKS_MIN = 0.55
#: The light under a match, against the lamp's own: below, a nearly flat patch
#: whose normalised score means nothing. A flash that switches during a
#: capture leaves a lamp at half its light, so half is well inside.
LIGHT_RANGE = (0.3, 2.0)
#: A lamp is not placed where another's symbol matches better by more than this.
LOCAL_MARGIN = 0.05
#: Hues further apart than this (OpenCV units, 0..180) are different lamp colours.
HUE_TOLERANCE = 18.0
#: A lamp found this close (px) to where it lit alone, all lit together and in
#: the bulb check has a fixed slot. A strip that closes up moves lamps by a
#: whole slot; the camera's noise moves a centre a fraction of a pixel.
FIXED_PX = 2.5
#: Of an element's box, the part that must change in a sweep state to count.
COVERAGE_MIN = 0.12
#: A shot whose lamp energy is below this share of its brightest shot is "off".
BLINK_RATIO = 0.45


# -- the plan ----------------------------------------------------------------

@dataclass
class LampStep:
    name: str
    set: dict[str, Any]
    blinks: bool = False


@dataclass
class SweepStep:
    name: str
    sweep: list[dict[str, Any]]


@dataclass
class TeachPlan:
    """What to switch on, and what each thing is called. See the HMI's ``can/teach_plan.json``."""

    lamps: list[LampStep]
    elements: list[SweepStep] = field(default_factory=list)
    rest: dict[str, Any] = field(default_factory=dict)
    settle_ms: int = 900
    dbc: Path | None = None
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TeachPlan":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        lamps = [LampStep(str(d["name"]), dict(d["set"]), bool(d.get("blinks", False)))
                 for d in raw.get("lamps", [])]
        elements = [SweepStep(str(d["name"]), [dict(s) for s in d["sweep"]])
                    for d in raw.get("elements", [])]
        names = [s.name for s in lamps] + [s.name for s in elements]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"{path.name}: names used twice: {', '.join(dupes)}")
        if not lamps and not elements:
            raise ValueError(f"{path.name} has no lamps and no elements to teach")
        dbc = raw.get("dbc")
        return cls(lamps=lamps, elements=elements, rest=dict(raw.get("rest", {})),
                   settle_ms=int(raw.get("settle_ms", 900)),
                   dbc=(path.parent / dbc) if dbc else None, path=path)

    def signals(self) -> set[str]:
        out = set(self.rest)
        for s in self.lamps:
            out |= set(s.set)
        for e in self.elements:
            for state in e.sweep:
                out |= set(state)
        return out

    def all_lamps(self) -> dict[str, Any]:
        """Every lamp's signals at once. Where two lamps set one signal, the later wins."""
        state: dict[str, Any] = {}
        for s in self.lamps:
            state.update(s.set)
        return state

    def steps(self) -> list[dict[str, Any]]:
        """The capture sequence: one entry per state, with how many shots to take.

        A flashing lamp is shot six times across 1.1 s, which catches it lit
        whatever the phase (the HMI's flash is 380 ms on, 380 ms off); the
        brightest of the shots is the lamp.
        """
        steps: list[dict[str, Any]] = [
            {"id": "rest", "kind": "rest", "name": "Rest", "set": dict(self.rest),
             "shots": 5, "gap_ms": 150}]
        for i, s in enumerate(self.lamps):
            steps.append({"id": f"lamp{i}", "kind": "lamp", "name": s.name,
                          "set": {**self.rest, **s.set},
                          "shots": 6 if s.blinks else 2, "gap_ms": 190 if s.blinks else 120})
        if len(self.lamps) > 1:
            steps.append({"id": "all", "kind": "all", "name": "All lamps",
                          "set": {**self.rest, **self.all_lamps()}, "shots": 6, "gap_ms": 190})
        for i, e in enumerate(self.elements):
            for j, state in enumerate(e.sweep):
                steps.append({"id": f"el{i}.{j}", "kind": "sweep", "name": e.name,
                              "set": {**self.rest, **state}, "shots": 2, "gap_ms": 120})
        steps.append({"id": "rest2", "kind": "rest", "name": "Rest again",
                      "set": dict(self.rest), "shots": 2, "gap_ms": 120})
        for s in steps:
            s["settle_ms"] = self.settle_ms
        return steps

    def check_steps(self) -> list[dict[str, Any]]:
        """A lamp check: every lamp alone, then all together."""
        return [s for s in self.steps() if s["kind"] in ("lamp", "all")] + [
            {"id": "rest2", "kind": "rest", "name": "Rest again", "set": dict(self.rest),
             "shots": 2, "gap_ms": 120, "settle_ms": self.settle_ms}]


# -- pixels ------------------------------------------------------------------

def _soft(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (3, 3), 0)


def lit_delta(baseline: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """How much each pixel lit up: brightening, or taking on a lens colour.

    Only brightening counts -- a lamp switching on is the change looked for.
    """
    base, cur = _soft(baseline), _soft(frame)
    dv = cur.max(axis=2).astype(np.float32) - base.max(axis=2).astype(np.float32)
    np.clip(dv, 0, None, out=dv)
    lb = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    lc = cv2.cvtColor(cur, cv2.COLOR_BGR2LAB).astype(np.float32)
    dc = np.linalg.norm(lc[..., 1:] - lb[..., 1:], axis=2)
    return np.maximum(dv, dc * CHROMA_WEIGHT)


def change(baseline: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """How much each pixel changed, either way: for elements, which go dark as well as light."""
    return cv2.absdiff(_soft(frame), _soft(baseline)).max(axis=2).astype(np.float32)


def noise_floor(frames: list[np.ndarray]) -> tuple[float, float]:
    """(lit, change): the most the rest screen differs from itself, over every pair of shots.

    Near the top of the difference, not its median: a handful of noisy pixels
    become a blob once strokes are joined.
    """
    lit = chg = 0.0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            lit = max(lit, float(np.percentile(lit_delta(frames[i], frames[j]), 99.95)))
            chg = max(chg, float(np.percentile(change(frames[i], frames[j]), 99.95)))
    return lit, chg


def _threshold(floor: float, minimum: float) -> float:
    return max(minimum, 1.3 * floor + 4.0)


@dataclass
class Blob:
    box: tuple[int, int, int, int]
    area: int
    strength: float
    mask: np.ndarray                 # bool, the frame's size

    @property
    def score(self) -> float:
        return self.strength * self.area

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0


def blobs(delta: np.ndarray, level: float, *, min_area: int = 12,
          within: np.ndarray | None = None) -> list[Blob]:
    """Groups of pixels above ``level``, the strokes of one symbol joined.

    Components close to each other -- a quarter of the larger's size apart --
    are one blob: an arrow and its tail, the letters inside a circle.
    """
    mask = (delta >= level).astype(np.uint8)
    if within is not None:
        mask &= within.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    boxes = {i: [int(v) for v in stats[i, :4]] for i in parts}
    group = {i: i for i in parts}

    def root(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    changed = True
    while changed:
        changed = False
        roots = sorted({root(i) for i in parts})
        extent = {}
        for r in roots:
            members = [i for i in parts if root(i) == r]
            x0 = min(boxes[i][0] for i in members)
            y0 = min(boxes[i][1] for i in members)
            x1 = max(boxes[i][0] + boxes[i][2] for i in members)
            y1 = max(boxes[i][1] + boxes[i][3] for i in members)
            extent[r] = (x0, y0, x1, y1)
        for a in roots:
            for b in roots:
                if a >= b or root(a) == root(b):
                    continue
                ax0, ay0, ax1, ay1 = extent[a]
                bx0, by0, bx1, by1 = extent[b]
                reach = max(4.0, 0.25 * max(ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0))
                if (ax0 - reach < bx1 and bx0 - reach < ax1
                        and ay0 - reach < by1 and by0 - reach < ay1):
                    group[root(b)] = root(a)
                    changed = True
    out = []
    for r in sorted({root(i) for i in parts}):
        m = np.isin(labels, [i for i in parts if root(i) == r])
        ys, xs = np.nonzero(m)
        box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
               int(ys.max() - ys.min() + 1))
        out.append(Blob(box, int(m.sum()), float(delta[m].mean()), m))
    return sorted(out, key=lambda b: -b.score)


def hue_signature(frame: np.ndarray, mask: np.ndarray) -> tuple[float | None, float]:
    """(mean hue, share of coloured pixels); hue None for a white lamp.

    Hue is averaged round the circle: red sits at both ends of OpenCV's 0..180.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total = int(mask.sum())
    if total == 0:
        return None, 0.0
    coloured = mask & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 60)
    ratio = float(coloured.sum()) / total
    if ratio < 0.25:
        return None, ratio
    a = hsv[..., 0][coloured].astype(np.float64) * (2 * math.pi / 180.0)
    mean = math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))
    return (math.degrees(mean) % 360.0) / 2.0, ratio


def hue_distance(a: float, b: float) -> float:
    gap = abs(a - b) % 180.0
    return min(gap, 180.0 - gap)


def hue_name(hue: float | None) -> str:
    if hue is None:
        return "white"
    for limit, name in ((8, "red"), (22, "amber"), (33, "yellow"), (85, "green"),
                        (100, "cyan"), (135, "blue"), (165, "magenta"), (180, "red")):
        if hue < limit:
            return name
    return "red"


def _centroid(delta: np.ndarray, box: tuple[int, int, int, int], level: float) -> tuple[float, float]:
    """Where the lit symbol's light is centred, sub-pixel: what its place is measured by."""
    x, y, w, h = box
    patch = np.clip(delta[y:y + h, x:x + w] - level * 0.5, 0, None)
    total = float(patch.sum())
    if total <= 0:
        return x + w / 2.0, y + h / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    return x + float((patch * xs).sum()) / total, y + float((patch * ys).sum()) / total


def _pad(box: tuple[int, int, int, int], frac: float, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    p = int(round(frac * max(w, h)))
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(shape[1], x + w + p), min(shape[0], y + h + p)
    return x0, y0, x1 - x0, y1 - y0


# -- what was learnt ---------------------------------------------------------

@dataclass
class Lamp:
    name: str
    set: dict[str, Any]
    blinks: bool
    box: tuple[int, int, int, int]
    centre: tuple[float, float]
    template: np.ndarray             # lit delta round the symbol, float32
    hue: float | None
    colour_ratio: float
    strength: float
    flashed: bool = False            # seen off in some shots while taught
    slot_all: tuple[float, float] | None = None
    order_all: int | None = None
    also: list[tuple[int, int, int, int]] = field(default_factory=list)
    offset: tuple[int, int] | None = None
    """Where the lamp's box starts inside its template. Padding round a lamp
    by the frame's edge is clipped, so it is not always the same each side."""
    states: dict[str, dict] = field(default_factory=dict)
    """Each state that lights it: label -> {"set": signals, "flashing": bool}.
    ABS On is steady, ABS Flash flashes: one lamp, two states."""
    slot_check: tuple[float, float] | None = None
    fixed: bool | None = None
    """Lights at one place whatever else is lit (None: not known yet)."""

    @property
    def energy(self) -> float:
        """The light the lamp adds, summed over its template."""
        return float(self.template.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "set": self.set, "blinks": self.blinks,
            "box": list(self.box), "centre": [round(c, 2) for c in self.centre],
            "colour": hue_name(self.hue),
            "hue": None if self.hue is None else round(self.hue, 1),
            "strength": round(self.strength, 1), "flashed": self.flashed,
            "slot_all": None if self.slot_all is None else [round(c, 2) for c in self.slot_all],
            "order_all": self.order_all,
            "also_changed": [list(b) for b in self.also],
            "offset": None if self.offset is None else list(self.offset),
            "states": self.states,
            "slot_check": None if self.slot_check is None else [round(c, 2) for c in self.slot_check],
            "fixed": self.fixed,
        }


@dataclass
class Detection:
    name: str                        # a lamp's name, or "unknown"
    box: tuple[int, int, int, int]
    centre: tuple[float, float]
    score: float
    hue: float | None
    flashing: bool | None = None
    lit_in: int = 1                  # shots it was lit in
    shot: int = -1                   # which shot, for a reading of one
    moved_px: float = 0.0            # how far its place wandered between those shots
    looks_like: str | None = None    # unknown: the lamp it is nearest to
    wrong_colour_of: str | None = None  # unknown: a lamp's symbol, in another colour

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "box": list(self.box),
                "centre": [round(c, 2) for c in self.centre],
                "match": round(self.score, 3), "colour": hue_name(self.hue),
                "flashing": self.flashing, "lit_in": self.lit_in,
                "moved_px": round(self.moved_px, 2), "looks_like": self.looks_like,
                "wrong_colour_of": self.wrong_colour_of}


@dataclass
class Library:
    """The lamps of one cluster, against the rest screen they were taught on."""

    baseline: np.ndarray
    lamps: list[Lamp]
    lit_floor: float
    change_floor: float
    lamp_size: tuple[float, float]
    zone: tuple[int, int, int, int] | None = None
    pitch: float | None = None       # centre to centre, all lit together

    @property
    def level(self) -> float:
        return _threshold(self.lit_floor, LIT_MIN)

    def get(self, name: str) -> Lamp | None:
        return next((lamp for lamp in self.lamps if lamp.name == name), None)

    def find(self, shots: list[np.ndarray]) -> list[Detection]:
        """Every lamp lit in these shots of one state, recognised by its symbol.

        Each shot is read on its own and the readings are then put together:
        a lamp lit in some shots and dark in others is flashing, and one whose
        place differs between shots moved -- a strip that reflows as a lamp
        flashes moves every lamp after it, and the shots stacked into one
        picture would show two layouts on top of each other. What is lit and
        matches no lamp's symbol well enough is "unknown".
        """
        readings = [self._find_one(lit_delta(self.baseline, s), s) for s in shots]
        for k, r in enumerate(readings):
            for d in r:
                d.shot = k
        merged: list[Detection] = []
        names = {d.name for r in readings for d in r if d.name != "unknown"}
        for name in names:
            ds = self._settled([d for r in readings for d in r if d.name == name])
            merged.append(self._merge(ds, len(shots)))
        unknowns = [d for r in readings for d in r if d.name == "unknown"]
        while unknowns:
            seed = unknowns.pop(0)
            same = [seed] + [d for d in unknowns if _iou(d.box, seed.box) > 0.3]
            unknowns = [d for d in unknowns if d not in same]
            merged.append(self._merge(same, len(shots)))
        return sorted(merged, key=lambda d: (round(d.centre[1] / max(1.0, self.lamp_size[1])),
                                             d.centre[0]))

    def _settled(self, ds: list[Detection]) -> list[Detection]:
        """One lamp's readings across shots, less those read elsewhere by mistake.

        A flashing lamp dark in a shot can be read onto another lamp that its
        own symbol fits less well -- one drawn over the gear letter, say -- at
        a clearly lower score than where it really is. Readings away from the
        best one's place and clearly weaker are dropped; one elsewhere that
        matches as well is a lamp that really moved, and stays.
        """
        if len(ds) < 2:
            return ds
        best = max(ds, key=lambda d: d.score)
        reach = 0.5 * max(self.lamp_size)
        return [d for d in ds
                if math.hypot(d.centre[0] - best.centre[0], d.centre[1] - best.centre[1]) <= reach
                or d.score >= best.score - LOCAL_MARGIN]

    def _merge(self, ds: list[Detection], shots: int) -> Detection:
        best = max(ds, key=lambda d: d.score)
        xs = [d.centre[0] for d in ds]
        ys = [d.centre[1] for d in ds]
        centre = (float(np.median(xs)), float(np.median(ys)))
        moved = max(math.hypot(x - centre[0], y - centre[1]) for x, y in zip(xs, ys))
        seen = {d.shot for d in ds}
        flashing = None if shots < 3 else turns([k in seen for k in range(shots)]) >= 2
        return Detection(best.name, best.box, centre, best.score, best.hue, flashing,
                         lit_in=len(ds), moved_px=moved, looks_like=best.looks_like,
                         wrong_colour_of=best.wrong_colour_of)

    def _find_one(self, delta: np.ndarray, shot: np.ndarray) -> list[Detection]:
        """The lamps lit in one shot: each lamp at most once, best matches first.

        A lit symbol no lamp matches to MATCH_MIN is "unknown", with the lamp
        it looks most like; one whose shape is a lamp's but whose colour is
        not carries that lamp's name as ``wrong_colour_of``.
        """
        level = self.level
        within = None
        if self.zone is not None:
            within = np.zeros(delta.shape, bool)
            x, y, w, h = self.zone
            within[y:y + h, x:x + w] = True
        lit = blobs(delta, level, min_area=max(12, int(0.08 * self.lamp_size[0] * self.lamp_size[1])),
                    within=within)
        # (score with colour, shape alone, lamp, box)
        candidates: list[tuple[float, float, Lamp, tuple[int, int, int, int]]] = []
        for b in lit:
            wx, wy, ww, wh = _pad(b.box, 0.6, delta.shape)
            window = delta[wy:wy + wh, wx:wx + ww]
            for lamp in self.lamps:
                th, tw = lamp.template.shape
                pad_y, pad_x = max(0, th - wh), max(0, tw - ww)
                window_p = (cv2.copyMakeBorder(window, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, 0)
                            if pad_y or pad_x else window)
                res = cv2.matchTemplate(window_p, lamp.template, cv2.TM_CCOEFF_NORMED)
                # Only where about as much light falls as the lamp gives: over
                # a nearly flat patch the normalised score saturates, and an
                # empty corner of the change map "matched" a lamp at 1.0.
                ii = cv2.integral(window_p.astype(np.float64))
                light = (ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw])
                light = light[:res.shape[0], :res.shape[1]] / max(lamp.energy, 1e-6)
                res = np.where((light >= LIGHT_RANGE[0]) & (light <= LIGHT_RANGE[1])
                               & np.isfinite(res), res, -1.0)
                peaks = (res >= LOOKS_MIN) & (res == cv2.dilate(
                    res, np.ones((max(3, th // 2) | 1, max(3, tw // 2) | 1), np.uint8)))
                lx, ly, lw, lh = lamp.box
                tx, ty, _, _ = _template_origin(lamp)
                for py, px in zip(*np.nonzero(peaks)):
                    box = _clip((int(wx + px + (lx - tx)), int(wy + py + (ly - ty)), lw, lh),
                                delta.shape)
                    shape = float(res[py, px])
                    candidates.append((_coloured(shape, lamp, box, delta, shot, level),
                                       shape, lamp, box))
        named = [c for c in candidates if c[0] >= MATCH_MIN]
        taken: list[Detection] = []
        used: set[str] = set()
        for score, _, lamp, box in sorted(named, key=lambda c: -c[0]):
            if lamp.name in used or any(_overlap(box, d.box) > 0.3 for d in taken):
                continue
            # Not where another lamp's symbol fits clearly better, even one
            # already placed elsewhere: a flashing seat-belt lamp in its dark
            # half otherwise settles on the airbag lamp beside it.
            if any(other > score + LOCAL_MARGIN for other, _, o_lamp, o_box in named
                   if o_lamp is not lamp and _overlap(o_box, box) > 0.5):
                continue
            # Nor where another lamp's symbol fits clearly better in shape and
            # only its colour is wrong: ABS drawn red is ABS in the wrong
            # colour, not the red brake lamp its circle resembles.
            if any(shape > score + LOCAL_MARGIN for coloured, shape, o_lamp, o_box in candidates
                   if coloured < 0 and o_lamp is not lamp and _overlap(o_box, box) > 0.5):
                continue
            x, y, w, h = box
            m = np.zeros(delta.shape, bool)
            m[y:y + h, x:x + w] = delta[y:y + h, x:x + w] >= level
            hue, _ = hue_signature(shot, m)
            used.add(lamp.name)
            taken.append(Detection(lamp.name, box, _centroid(delta, box, level), score, hue))
        # What is lit and no lamp accounts for.
        covered = np.zeros(delta.shape, bool)
        for d in taken:
            x, y, w, h = _pad(d.box, 0.1, delta.shape)
            covered[y:y + h, x:x + w] = True
        min_unknown = 0.25 * self.lamp_size[0] * self.lamp_size[1]
        for b in lit:
            rest = b.mask & ~covered
            if rest.sum() < 0.5 * b.area:
                continue
            ys, xs = np.nonzero(rest)
            box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                   int(ys.max() - ys.min() + 1))
            if box[2] * box[3] < min_unknown:
                continue
            hue, _ = hue_signature(shot, rest)
            d = Detection("unknown", box, _centroid(delta, box, level), 0.0, hue)
            near = [c for c in candidates if _overlap(c[3], box) > 0.5]
            if near:
                best = max(near, key=lambda c: c[0])
                if best[0] > 0:
                    d.looks_like, d.score = best[2].name, best[0]
                shaped = max(near, key=lambda c: c[1])
                if shaped[0] < 0 and shaped[1] >= MATCH_MIN:
                    d.wrong_colour_of = shaped[2].name
                    d.score = shaped[1]
            taken.append(d)
        return taken

    # -- persistence -----------------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "lamps-baseline.png"), self.baseline)
        np.savez_compressed(directory / "lamp-templates.npz",
                            **{f"t{i}": lamp.template for i, lamp in enumerate(self.lamps)})
        target = directory / "lamps.json"
        target.write_text(json.dumps({
            "lit_floor": self.lit_floor, "change_floor": self.change_floor,
            "lamp_size": list(self.lamp_size), "zone": None if self.zone is None else list(self.zone),
            "pitch": self.pitch, "lamps": [lamp.to_dict() for lamp in self.lamps],
        }, indent=2))
        return target

    @classmethod
    def load(cls, directory: Path) -> "Library":
        raw = json.loads((directory / "lamps.json").read_text())
        templates = np.load(directory / "lamp-templates.npz")
        lamps = []
        for i, d in enumerate(raw["lamps"]):
            lamps.append(Lamp(
                name=d["name"], set=d["set"], blinks=d["blinks"], box=tuple(d["box"]),
                centre=tuple(d["centre"]), template=templates[f"t{i}"].astype(np.float32),
                hue=d["hue"], colour_ratio=0.0, strength=d["strength"], flashed=d["flashed"],
                slot_all=None if d["slot_all"] is None else tuple(d["slot_all"]),
                order_all=d["order_all"], also=[tuple(b) for b in d["also_changed"]],
                offset=None if d.get("offset") is None else tuple(d["offset"]),
                states=d.get("states", {}),
                slot_check=None if d.get("slot_check") is None else tuple(d["slot_check"]),
                fixed=d.get("fixed")))
        return cls(baseline=cv2.imread(str(directory / "lamps-baseline.png")), lamps=lamps,
                   lit_floor=raw["lit_floor"], change_floor=raw["change_floor"],
                   lamp_size=tuple(raw["lamp_size"]),
                   zone=None if raw["zone"] is None else tuple(raw["zone"]), pitch=raw["pitch"])


def _coloured(score: float, lamp: Lamp, box, delta: np.ndarray, shot: np.ndarray,
              level: float) -> float:
    """A match's score with its colour taken into account.

    The same symbol in another colour is another lamp -- cruise control set
    (green) and ready (white) share one -- so a hue far from the lamp's rules
    it out, and coloured against white costs it a lot.
    """
    x, y, w, h = box
    m = np.zeros(delta.shape, bool)
    m[y:y + h, x:x + w] = delta[y:y + h, x:x + w] >= level
    hue, _ = hue_signature(shot, m)
    if lamp.hue is not None and hue is not None:
        return score if hue_distance(hue, lamp.hue) <= HUE_TOLERANCE else -1.0
    if (lamp.hue is None) != (hue is None):
        return score - 0.3
    return score


def _template_origin(lamp: Lamp) -> tuple[int, int, int, int]:
    """The template's box: the lamp's box padded as when it was cut."""
    x, y, w, h = lamp.box
    th, tw = lamp.template.shape
    ox, oy = lamp.offset if lamp.offset is not None else ((tw - w) // 2, (th - h) // 2)
    return x - ox, y - oy, tw, th


def _clip(box, shape) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(shape[1], x + w), min(shape[0], y + h)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def _overlap(a, b) -> float:
    """The part of the smaller box that the other covers."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / float(max(1, min(aw * ah, bw * bh)))


def _flashing(deltas: list[np.ndarray], box, level: float) -> bool | None:
    """Lit and dark by turns across the shots: None when there were too few to say."""
    if len(deltas) < 3:
        return None
    x, y, w, h = _clip(box, deltas[0].shape)
    energy = [float(np.clip(d[y:y + h, x:x + w] - level, 0, None).sum()) for d in deltas]
    top = max(energy)
    if top <= 0:
        return False
    return turns([e >= BLINK_RATIO * top for e in energy]) >= 2


def turns(lit: list[bool]) -> int:
    """How often a lamp went on or off across shots taken in order.

    A flash does so at least twice in the shots of a flashing state, which
    span more than one of its periods; a lamp that comes on during the first
    shot, or goes off during the last, does so once and is not flashing.
    """
    return sum(1 for a, b in zip(lit, lit[1:]) if a != b)


# -- teaching ----------------------------------------------------------------

def _median(shots: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(shots), axis=0).astype(np.uint8) if len(shots) > 1 else shots[0]


def _candidate(b: Blob, deltas: list[np.ndarray], shots: list[np.ndarray],
               level: float) -> dict[str, Any]:
    """What a lamp step keeps of one lit blob.

    The symbol comes from the one shot it is brightest in, not the stack of
    them all, and that shot is judged by the blob's own box: over the whole
    frame a camera's noise and backlight banding outweigh one small lamp, and
    a flashing lamp was learnt from a shot it was dark in.
    """
    bx, by, bw, bh = b.box
    k = int(np.argmax([float(np.clip(d[by:by + bh, bx:bx + bw] - level, 0, None).sum())
                       for d in deltas]))
    delta = deltas[k]
    lit = np.zeros(delta.shape, bool)
    lit[by:by + bh, bx:bx + bw] = delta[by:by + bh, bx:bx + bw] >= level
    x, y, w, h = _pad(b.box, 0.25, delta.shape)
    hue, ratio = hue_signature(shots[k], lit & b.mask)
    return {
        "box": b.box, "area": b.area, "score": b.score, "strength": b.strength,
        "mid": b.centre, "centre": _centroid(delta, b.box, level),
        "template": delta[y:y + h, x:x + w].copy(), "offset": (bx - x, by - y),
        "hue": hue, "ratio": ratio,
        "flashed": bool(_flashing(deltas, b.box, level)) if len(deltas) >= 3 else False,
    }


def learn(plan: TeachPlan, captured: dict[str, list[np.ndarray]]) -> tuple[Library, dict[str, Any]]:
    """Everything the teach-in captured, turned into lamps and named regions.

    ``captured`` maps each step id of ``plan.steps()`` to its shots. Returns
    the lamp library and a summary: per lamp, per element sweep, and what
    went wrong where (a step that changed nothing, a lamp found twice).
    """
    rest_shots = captured["rest"]
    baseline = _median(rest_shots)
    lit_floor, change_floor = noise_floor(rest_shots)
    level = _threshold(lit_floor, LIT_MIN)
    problems: list[str] = []

    # Pass one: every lamp step's lit blobs, each kept as what is needed of
    # it -- its box, and its symbol cut from the shot it is brightest in --
    # rather than the step's frames: a teach-in is a couple of hundred shots.
    found: dict[int, list[dict[str, Any]]] = {}
    for i, step in enumerate(plan.lamps):
        shots = captured.get(f"lamp{i}")
        if not shots:
            problems.append(f"{step.name}: not captured")
            continue
        deltas = [lit_delta(baseline, s) for s in shots]
        bs = blobs(np.max(deltas, axis=0), level)
        if not bs:
            problems.append(f"{step.name}: nothing lit up -- the signal did not take "
                            "effect, or the lamp is dimmer than the screen's own noise")
            continue
        found[i] = [_candidate(b, deltas, shots, level) for b in bs]
        del deltas, shots

    # A lamp's size, from the steps where one thing lit and nothing else.
    singles = [cs[0]["box"] for cs in found.values() if len(cs) == 1] or \
              [cs[0]["box"] for cs in found.values()]
    tw = float(np.median([b[2] for b in singles])) if singles else 32.0
    th = float(np.median([b[3] for b in singles])) if singles else 32.0
    single_centres = [cs[0]["mid"] for cs in found.values() if len(cs) == 1]

    lamps: list[Lamp] = []
    for i, cs in found.items():
        step = plan.lamps[i]

        def lampness(c: dict[str, Any]) -> float:
            # Sized like the other lamps, and near where they light.
            size = abs(math.log(c["box"][2] / tw)) + abs(math.log(c["box"][3] / th))
            near = min((math.hypot(c["mid"][0] - m[0], c["mid"][1] - m[1])
                        for m in single_centres if m != c["mid"]), default=0.0)
            return size + 0.1 * min(near / max(tw, th), 10.0)

        big = [c for c in cs if c["area"] >= 0.15 * tw * th] or cs
        best = min(big, key=lambda c: (round(lampness(c), 1), -c["score"]))
        lamps.append(Lamp(
            name=step.name, set=step.set, blinks=step.blinks, box=best["box"],
            centre=best["centre"], template=best["template"], hue=best["hue"],
            colour_ratio=best["ratio"], strength=best["strength"], flashed=best["flashed"],
            also=[c["box"] for c in cs if c is not best], offset=best["offset"]))

    lib = Library(baseline=baseline, lamps=lamps, lit_floor=lit_floor,
                  change_floor=change_floor, lamp_size=(tw, th))
    _set_zone(lib)

    # All lit together: each lamp's slot, and the strip's order and pitch.
    all_report: list[dict[str, Any]] = []
    if "all" in captured and lamps:
        all_report = learn_all(lib, captured["all"], problems)

    # Elements: what each sweep changes, state by state.
    sweeps: list[dict[str, Any]] = []
    change_level = _threshold(change_floor, CHANGE_MIN)
    for i, e in enumerate(plan.elements):
        states = []
        for j in range(len(e.sweep)):
            shots = captured.get(f"el{i}.{j}")
            if not shots:
                continue
            d = change(baseline, _median(shots))
            m = cv2.morphologyEx((d >= change_level).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8))
            states.append(m.astype(bool))
        union = np.any(states, axis=0) if states else None
        boxes = []
        if union is not None and union.any():
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)))
            boxes = [tuple(int(v) for v in stats[k, :4]) for k in range(1, n)
                     if stats[k, cv2.CC_STAT_AREA] >= 30]
        if not boxes:
            problems.append(f"{e.name}: nothing changed (not drawn on this screen?)")
        sweeps.append({"name": e.name, "states": states, "regions": boxes})

    # Back at rest, the screen should be as it was.
    if "rest2" in captured:
        d = change(baseline, _median(captured["rest2"]))
        moved = float((d >= change_level).mean())
        if moved > 0.002:
            problems.append(f"back at rest, {100 * moved:.1f}% of the screen differs from "
                            "the start: the camera, the light or the exposure changed "
                            "during the teach-in")

    summary = {
        "lamps": [lamp.to_dict() for lamp in lamps],
        "layout": layout(lib),
        "all_lit": all_report,
        "lamp_size": [round(tw, 1), round(th, 1)],
        "pitch": None if lib.pitch is None else round(lib.pitch, 2),
        "floors": {"lit": round(lit_floor, 2), "change": round(change_floor, 2)},
        "elements": [{"name": s["name"], "regions": [list(b) for b in s["regions"]]}
                     for s in sweeps],
        "problems": problems,
    }
    summary["_sweeps"] = sweeps            # for naming; not serialised
    return lib, summary


def learn_all(lib: Library, shots: list[np.ndarray], problems: list[str]) -> list[dict[str, Any]]:
    """Every lamp lit at once: each lamp's slot, and the strip's order and pitch.

    Looked for everywhere: lit together they may spread well past where any
    one of them lights alone -- every lamp alone lights in a centred strip's
    middle slot.
    """
    lib.zone = None
    dets = lib.find(shots)
    named = [d for d in dets if d.name != "unknown"]
    for order, d in enumerate(named):
        lamp = lib.get(d.name)
        if lamp is not None and lamp.slot_all is None:
            lamp.slot_all, lamp.order_all = d.centre, order
    seen = [d.name for d in named]
    twice = sorted({n for n in seen if seen.count(n) > 1})
    if twice:
        problems.append(f"all lamps: {', '.join(twice)} recognised more than once")
    wandered = [f"{d.name} ({d.moved_px:.0f} px)" for d in named if d.moved_px > 2.0]
    if wandered:
        problems.append("all lamps: moved between shots, as a strip that reflows "
                        "while a lamp flashes does: " + ", ".join(wandered))
    unknown = [d for d in dets if d.name == "unknown"]
    if unknown:
        problems.append(f"all lamps: {len(unknown)} lit symbol(s) matched no lamp")
    lib.pitch = _pitch([d.centre for d in named], lib.lamp_size[1])
    # Where lamps light, from lamps: what else changed with them (a set
    # speed, the lane graphic) is not a place a lamp is looked for.
    _set_zone(lib, extra=[d.box for d in named])
    _settle_fixed(lib)
    return [d.to_dict() for d in dets]


def learn_bulb_check(lib: Library, shots: list[np.ndarray], problems: list[str]) -> list[dict[str, Any]]:
    """The bulb check at power-up: where each lamp sits when every one is lit.

    A production cluster lights every lamp for a few seconds when the
    ignition comes on, lamps that can never be lit together over CAN
    included. Their places then are what a later bulb check is held to.
    """
    zone, lib.zone = lib.zone, None
    dets = lib.find(shots)
    named = [d for d in dets if d.name != "unknown"]
    for d in named:
        lamp = lib.get(d.name)
        if lamp is not None and lamp.slot_check is None:
            lamp.slot_check = d.centre
    missing = [lamp.name for lamp in lib.lamps if lamp.slot_check is None]
    if missing:
        problems.append(f"bulb check: not lit -- {', '.join(missing)}")
    lib.zone = zone
    _set_zone(lib, extra=[d.box for d in named])
    _settle_fixed(lib)
    return [d.to_dict() for d in dets]


def _settle_fixed(lib: Library) -> None:
    """Which lamps have a slot of their own: the same place in every state seen.

    One lamp that moves means a strip that closes up, and then no lamp is
    taken to be fixed: in a centred strip of seven, the middle lamp lights in
    the middle slot both alone and with all lit, by coincidence. A lamp seen
    in one state only stays unknown (None), and is checked as a lamp of a
    strip would be.
    """
    stays: dict[str, bool] = {}
    for lamp in lib.lamps:
        places = [p for p in (lamp.slot_all, lamp.slot_check) if p is not None]
        if places:
            stays[lamp.name] = all(math.hypot(p[0] - lamp.centre[0], p[1] - lamp.centre[1])
                                   <= FIXED_PX for p in places)
    closes_up = not all(stays.values())
    for lamp in lib.lamps:
        lamp.fixed = None if lamp.name not in stays else (stays[lamp.name] and not closes_up)


def layout(lib: Library) -> str:
    """"fixed slots", "closes up" (a strip of only the lit lamps) or "unknown"."""
    known = [lamp.fixed for lamp in lib.lamps if lamp.fixed is not None]
    if not known:
        return "unknown"
    return "fixed slots" if all(known) else "closes up"


def _set_zone(lib: Library, extra: list[tuple[int, int, int, int]] = ()) -> None:
    """Where lamps light: every lamp's box, padded by a lamp's size."""
    boxes = [lamp.box for lamp in lib.lamps] + list(extra)
    if not boxes:
        return
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    p = int(round(max(lib.lamp_size)))
    h, w = lib.baseline.shape[:2]
    lib.zone = (max(0, x0 - p), max(0, y0 - p), min(w, x1 + p) - max(0, x0 - p),
                min(h, y1 + p) - max(0, y0 - p))


def _pitch(centres: list[tuple[float, float]], row_height: float) -> float | None:
    """Centre-to-centre spacing of neighbours in a row, typical."""
    if len(centres) < 2:
        return None
    rows: dict[int, list[float]] = {}
    for cx, cy in centres:
        rows.setdefault(int(round(cy / max(1.0, row_height))), []).append(cx)
    gaps = [b - a for xs in rows.values() for a, b in zip(sorted(xs), sorted(xs)[1:])]
    return float(np.median(gaps)) if gaps else None


def name_elements(profile, sweeps: list[dict[str, Any]]) -> dict[str, str]:
    """Each element of ``profile`` named by the sweep that changes it, or "unknown".

    An element belongs to a sweep when its box changes in most of that
    sweep's states; between sweeps, the one changing most of it wins. The
    names are made unique by number, left to right, top to bottom.
    """
    names: dict[str, str] = {}
    for spec in profile:
        x, y, w, h = (int(round(v)) for v in spec.bbox)
        best, best_cover = None, 0.0
        for s in sweeps:
            states = s["states"]
            if not states:
                continue
            covers = []
            for m in states:
                box = m[max(0, y):y + h, max(0, x):x + w]
                covers.append(float(box.mean()) if box.size else 0.0)
            hits = sum(c >= COVERAGE_MIN for c in covers)
            if hits * 2 > len(states) and np.mean(covers) > best_cover:
                best, best_cover = s["name"], float(np.mean(covers))
        names[spec.id] = best or "unknown"
    return names


def rename(profile, names: dict[str, str]) -> dict[str, str]:
    """Give the profile's elements their names: old id -> new id.

    "Speed" for one, "Speed 1", "Speed 2" ... in reading order for several;
    an unknown one keeps its place in its id: ``unknown@312,75``.
    """
    groups: dict[str, list] = {}
    for spec in profile:
        groups.setdefault(names.get(spec.id, "unknown"), []).append(spec)
    mapping = {}
    for name, specs in groups.items():
        specs.sort(key=lambda s: (s.bbox[1] // 20, s.bbox[0]))
        for k, spec in enumerate(specs, 1):
            if name == "unknown":
                new = f"unknown@{int(spec.bbox[0])},{int(spec.bbox[1])}"
            else:
                new = name if len(specs) == 1 else f"{name} {k}"
            mapping[spec.id] = new
    for spec in profile:
        spec.id = mapping[spec.id]
    return mapping


# -- checking ----------------------------------------------------------------

@dataclass
class LampResult:
    name: str
    verdict: str                     # PASS / REVIEW / FAIL
    reason: str | None
    expected: bool
    found: bool
    centre: tuple[float, float] | None = None
    dx: float | None = None
    dy: float | None = None
    colour: str | None = None
    flashing: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "verdict": self.verdict, "reason": self.reason,
             "expected": self.expected, "found": self.found, "colour": self.colour,
             "flashing": self.flashing}
        if self.centre is not None:
            d["centre"] = [round(c, 2) for c in self.centre]
        if self.dx is not None:
            d["dx"], d["dy"] = round(self.dx, 2), round(self.dy, 2)
            d["delta_px"] = round(math.hypot(self.dx, self.dy), 2)
        return d


def check(lib: Library, found: list[Detection], expected: list[str], *,
          where: str = "", tol_warn: float = 1.5, tol_fail: float = 2.5,
          flashing: dict[str, bool] | None = None) -> list[LampResult]:
    """These detections against the lamps that should be lit.

    A lamp with a fixed slot is expected in its slot, to the tolerance,
    whatever else is lit. Any other lamp is expected where it was taught when
    ``where`` names that state exactly ("alone", "all" lit together, the bulb
    "check"); in any other state a strip that closes up may lay it out
    differently, so only that it is lit, its colour and its order along the
    strip are checked.
    """
    out: list[LampResult] = []
    by_name: dict[str, list[Detection]] = {}
    for d in found:
        by_name.setdefault(d.name, []).append(d)
    recoloured = {d.wrong_colour_of: d for d in by_name.get("unknown", []) if d.wrong_colour_of}
    for name in expected:
        lamp = lib.get(name)
        ds = by_name.get(name, [])
        if not ds and name in recoloured:
            d = recoloured[name]
            by_name["unknown"].remove(d)
            out.append(LampResult(name, "FAIL", "wrong_colour", True, True, d.centre,
                                  colour=hue_name(d.hue), flashing=d.flashing))
            continue
        if not ds:
            out.append(LampResult(name, "FAIL", "missing", True, False))
            continue
        d = ds[0]
        r = LampResult(name, "PASS", None, True, True, d.centre, colour=hue_name(d.hue),
                       flashing=d.flashing)
        want = None
        if lamp is not None and lamp.fixed:
            want = lamp.centre
        elif lamp is not None and where == "alone":
            want = lamp.centre
        elif lamp is not None and where == "all":
            want = lamp.slot_all
        elif lamp is not None and where == "check":
            want = lamp.slot_check
        if want is not None:
            r.dx, r.dy = d.centre[0] - want[0], d.centre[1] - want[1]
            off = math.hypot(r.dx, r.dy)
            if off > tol_fail:
                r.verdict, r.reason = "FAIL", "position"
            elif off > tol_warn:
                r.verdict, r.reason = "REVIEW", "position_marginal"
        if d.moved_px > tol_fail:
            r.verdict, r.reason = "FAIL", r.reason or "moves_between_shots"
        blinks = (flashing or {}).get(name, lamp.blinks if lamp is not None else None)
        if where == "check":
            blinks = False                            # the bulb check holds every lamp steady
        if blinks and d.flashing is False:
            r.verdict, r.reason = _worse(r.verdict, "REVIEW"), r.reason or "not_flashing"
        if blinks is False and d.flashing:
            r.verdict, r.reason = _worse(r.verdict, "REVIEW"), r.reason or "flashing"
        out.append(r)
        for extra in ds[1:]:
            out.append(LampResult(name, "FAIL", "shown_twice", True, True, extra.centre,
                                  colour=hue_name(extra.hue)))
    side = [b for name in expected if lib.get(name) is not None for b in lib.get(name).also]
    # An expected lamp's own place: a symbol there that matches nothing is
    # that lamp at another brightness -- one that fades rather than flashes
    # (the HMI's shift prompt, 15% to full) matches its own symbol only when
    # bright enough. A wrong symbol in its place still leaves it missing.
    own = [d.box for name in expected for d in by_name.get(name, [])]
    own += [lib.get(name).box for name in expected
            if lib.get(name) is not None and lib.get(name).fixed]
    for name, ds in by_name.items():
        if name in expected:
            continue
        for d in ds:
            if name == "unknown" and any(_overlap(d.box, b) > 0.3 for b in side):
                continue                  # what the expected lamp's signals also change
            if name == "unknown" and any(_overlap(d.box, b) > 0.3 for b in own):
                continue                  # the expected lamp, dim
            reason = "unknown_symbol" if name == "unknown" else "unexpected"
            if name == "unknown" and d.wrong_colour_of:
                reason = f"unknown_symbol: {d.wrong_colour_of}'s symbol in {hue_name(d.hue)}"
            elif name == "unknown" and d.looks_like:
                reason = f"unknown_symbol: nearest is {d.looks_like} ({d.score:.2f})"
            out.append(LampResult(name, "FAIL", reason, False, True, d.centre,
                                  colour=hue_name(d.hue), flashing=d.flashing))
    # Order along the strip, as when all were lit: lamps without a slot of
    # their own (one with a slot is already held to its place).
    shown = [r for r in out if r.expected and r.found and r.reason != "shown_twice"]
    ranked = [(lib.get(r.name).order_all, r) for r in shown
              if lib.get(r.name) is not None and lib.get(r.name).order_all is not None
              and not lib.get(r.name).fixed]
    if len(ranked) >= 2:
        row = max(1.0, lib.lamp_size[1])
        seen = sorted(ranked, key=lambda t: (round(t[1].centre[1] / row), t[1].centre[0]))
        keep = _longest_rising([t[0] for t in seen])
        for k, (_, r) in enumerate(seen):
            if k not in keep:
                r.verdict, r.reason = "FAIL", r.reason or "order"
    return out


def _longest_rising(values: list[int]) -> set[int]:
    """Indices of the longest increasing run (not necessarily contiguous).

    What is out of order is what falls outside it: two lamps swapped put one
    of them out, not the whole strip.
    """
    best: list[list[int]] = []
    for i, v in enumerate(values):
        prev = [b for b in best if values[b[-1]] < v]
        chain = (max(prev, key=len) if prev else []) + [i]
        best.append(chain)
    return set(max(best, key=len)) if best else set()


def _worse(a: str, b: str) -> str:
    rank = {"FAIL": 0, "REVIEW": 1, "PASS": 2}
    return a if rank[a] <= rank[b] else b


def verdict(results: list[LampResult]) -> str:
    return min((r.verdict for r in results), key=lambda v: {"FAIL": 0, "REVIEW": 1, "PASS": 2}[v],
               default="PASS")


# -- pictures ----------------------------------------------------------------

_COLOURS = {"PASS": (60, 200, 60), "REVIEW": (0, 190, 255), "FAIL": (60, 60, 235)}


def draw_lamps(frame: np.ndarray, found: list[Detection],
               results: list[LampResult] | None = None) -> np.ndarray:
    """Each detection boxed and named; coloured by its result when there is one."""
    out = frame.copy()
    verdicts = {}
    for r in results or []:
        verdicts.setdefault(r.name, r.verdict)
    for d in found:
        colour = _COLOURS.get(verdicts.get(d.name, "PASS" if d.name != "unknown" else "FAIL"))
        x, y, w, h = d.box
        cv2.rectangle(out, (x - 2, y - 2), (x + w + 1, y + h + 1), colour, 2)
        _label(out, d.name, (x, y - 6), colour)
    for r in results or []:
        if r.reason == "missing":
            _label(out, f"missing: {r.name}", (8, 22 + 18 * sum(
                1 for q in results if q.reason == "missing" and q is not r and
                results.index(q) < results.index(r))), _COLOURS["FAIL"])
    return out


def draw_elements(frame: np.ndarray, profile) -> np.ndarray:
    """The rest screen with every element boxed: named in green, unknown in grey."""
    out = frame.copy()
    for spec in profile:
        x, y, w, h = (int(round(v)) for v in spec.bbox)
        unknown = spec.id.startswith("unknown@")
        colour = (150, 150, 150) if unknown else (60, 200, 60)
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 1 if unknown else 2)
        if not unknown:
            _label(out, spec.id, (x, y - 5), colour)
    return out


def _label(img: np.ndarray, text: str, at: tuple[int, int], colour) -> None:
    x, y = at
    y = max(12, y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

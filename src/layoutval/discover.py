"""What every signal shows on the cluster, found from the DBC alone.

No teach plan: every signal the cluster receives is tried. A bit is flipped
from its rest value; a value table is tried value by value (but not "SNA",
"Init" or "Invalid", which mean the signal is broken); a number is swept to
three points across its range. Each state is shot, and compared with the
rest screen shot just before it:

* a compact symbol lighting up where the screen was dark, sized like the
  other lamps, from a bit or a value table, is a **lamp**. States of one
  signal that light the same symbol in the same place are one lamp with a
  behaviour each -- ABS On steady, ABS Flash flashing;
* a number lights a lamp too, at one end of its range -- low fuel, coolant
  hot. Swept, it changes its gauge and may light a symbol: that symbol is a
  lamp when it lights in one of the bulb check's places ("Fuel level low"),
  since a readout's digits also come and go in dark places;
* a change to what was already drawn is an **element** ("Vehicle speed",
  "Selected gear"), named on the rest screen as :func:`lamps.name_elements`
  does;
* what still differs after the signal is back at rest changed with time,
  not with the signal -- the odometer and the trip counting while the speed
  was swept -- and is not put down to it;
* a signal whose states change nothing is listed as such: on some clusters
  it needs another signal set first (the gear engaged shows only in D), and
  ``rig.json`` can say so with ``"when"``.

Names come from ``rig.json``, else the DBC's comment on the signal, else the
signal's own name without its sender and its ``TtReq``/``Sts`` endings.

The bulb check at power-up (every lamp lit) says where all the lamps are, so
a lamp no signal lit is still found: it is reported as an unknown lamp, with
its place (see :func:`layoutval.lamps.learn_bulb_check`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from layoutval import lamps as L
from layoutval.canbus import NOT_AVAILABLE, Rig

#: Where across a number's range it is swept to. Both ends, where warnings are
#: (low fuel, coolant hot), and low in the range, where a gauge's scale
#: usually is: a DBC's range is often the raw field's, far past any dial.
SWEEP_POINTS = (0.05, 0.25, 0.6, 0.95)
#: Values of a value table tried, at most.
MAX_VALUES = 8
#: A lamp is this size of the typical lamp, at most and at least, each way.
LAMP_SIZE_RANGE = (0.5, 2.0)
#: Rest brightness (V) under which the screen counts as dark there.
DARK = 70
#: Two states' symbols this alike, in one place and colour, are one lamp. Two
#: learnt templates are both noisy, so they match each other less well than
#: either matches a frame: the HMI's brake lamp On and Flash, through the
#: page and the simulated webcam, matched at 0.78; a different symbol in the
#: same place does not come near.
SAME_LAMP = 0.6
#: ...and this close, as a share of a lamp's size.
SAME_PLACE = 0.35
#: A change smaller than this many pixels is a speck of noise, not a change.
MIN_CHANGE_PX = 12

ENDINGS = re.compile(r"(_?(tt_?req|req|rq|request|sts|stat|status|state|ind|cmd|disp|dsp|"
                     r"actl|act|lamp|lmp|tt|telltale)|_(b|d|no))+$", re.I)


@dataclass
class Probe:
    signal: str
    value: Any
    label: str
    set: dict[str, Any]


@dataclass
class SignalProbes:
    signal: str
    name: str
    kind: str                       # "state" (bits, value tables) or "sweep" (numbers)
    probes: list[Probe] = field(default_factory=list)
    rest: float | None = None       # a number's rest value


def display_name(signal, rig: Rig, nodes: set[str], value: str | None = None) -> str:
    """What to call what a signal shows: rig.json, the DBC's comment, or its name."""
    if value is not None and f"{signal.name}={value}" in rig.names:
        return rig.names[f"{signal.name}={value}"]
    if signal.name in rig.names:
        return rig.names[signal.name]
    comment = (signal.comment or "").strip()
    if comment:
        first = re.split(r"[.:;]\s|[.:;]$|\n", comment)[0].strip()
        first = re.sub(r"\s+(request|req\.?)$", "", first, flags=re.I)
        if 2 <= len(first) <= 40:
            return first[0].upper() + first[1:]
    return humanise(signal.name, nodes)


def humanise(name: str, nodes: set[str]) -> str:
    """ESC_ABS_TtReq -> ABS; BCM_HighBeam_TtReq -> High beam; SRS_BuckleDriver -> Buckle driver."""
    parts = name.split("_")
    if len(parts) > 1 and (parts[0] in nodes or parts[0].isupper() and len(parts[0]) <= 5):
        parts = parts[1:]
    core = "_".join(parts)
    stripped = ENDINGS.sub("", core) or core
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", stripped.replace("_", " ")).split()
    out = []
    for w in words:
        out.append(w if w.isupper() and len(w) > 1 else w.lower())
    text = " ".join(out) or name
    return text[0].upper() + text[1:]


class Discovery:
    """Every signal the cluster receives, and the states to try it in."""

    def __init__(self, rig: Rig, db=None) -> None:
        import cantools

        self.rig = rig
        self.db = db or cantools.database.load_file(str(rig.dbc))
        self.nodes = {n.name for n in self.db.nodes} | {
            s for m in self.db.messages for s in (m.senders or [])}
        self.signals = self._candidates()
        self.path = rig.dbc
        self.lamps: list = []           # TeachPlan compatibility: filled by learn()
        self.elements: list = []
        self.settle_ms = 0

    # -- what to try -----------------------------------------------------------

    def _candidates(self) -> list[SignalProbes]:
        rig = self.rig
        e2e = {n for e in rig.e2e.values() for n in (e.counter, e.crc) if n}
        skip = set(rig.exclude) | e2e
        if rig.ignition:
            skip.add(rig.ignition["signal"])
        when = getattr(rig, "when", None) or {}
        out = []
        for m in self.db.messages:
            if m.name not in rig.send or m.name in rig.exclude:
                continue
            for s in m.signals:
                if s.name in skip or s.length > 32:
                    continue
                if rig.cluster_node and s.receivers and rig.cluster_node not in s.receivers:
                    continue
                probes = self._probes(s)
                if not probes:
                    continue
                cond = dict(when.get(s.name, {}))
                for p in probes:
                    p.set = {**cond, s.name: p.value}
                kind = "sweep" if self._numeric(s) else "state"
                rest = float(s.raw_to_scaled(self._rest_raw(s))) if kind == "sweep" else None
                out.append(SignalProbes(s.name, display_name(s, rig, self.nodes), kind, probes,
                                        rest))
        return out

    @staticmethod
    def _numeric(s) -> bool:
        choices = {int(k): str(v) for k, v in (s.choices or {}).items()}
        real = [v for v in choices.values() if not NOT_AVAILABLE.search(v)]
        return s.length > 1 and not real

    def _rest_raw(self, s) -> int:
        value = self.rig.rest.get(s.name)
        if value is None:
            return int(s.raw_initial or 0)
        if isinstance(value, str):
            return int(s.choice_to_number(value))
        return int(round(float(s.scaled_to_raw(float(value)))))

    def _probes(self, s) -> list[Probe]:
        rest = self._rest_raw(s)
        choices = {int(k): str(v) for k, v in (s.choices or {}).items()}
        if not self._numeric(s):
            if choices:
                values = [(name, name) for raw, name in choices.items()
                          if raw != rest and not NOT_AVAILABLE.search(name)][:MAX_VALUES]
            else:
                values = [(1 - min(1, rest), str(1 - min(1, rest)))]
            return [Probe(s.name, v, label, {}) for v, label in values]
        lo = s.minimum if s.minimum is not None else float(s.raw_to_scaled(0))
        hi = s.maximum if s.maximum is not None else float(s.raw_to_scaled((1 << s.length) - 2))
        if hi <= lo:
            return []
        sna = {raw for raw, name in choices.items() if NOT_AVAILABLE.search(name)}
        rest_value = float(s.raw_to_scaled(rest))
        # Swept away from the end the rest value is nearer, so that it comes
        # back to rest from the far side: a warning with hysteresis (low fuel
        # on at 12 %, off at 15 %) lit at the near end stays lit coming back
        # to a rest value inside its band, and the screen would not be at
        # rest again.
        points = SWEEP_POINTS if (rest_value - lo) <= (hi - rest_value) else SWEEP_POINTS[::-1]
        out = []
        for f in points:
            v = lo + f * (hi - lo)
            raw = int(round(float(s.scaled_to_raw(v))))
            v = float(s.raw_to_scaled(raw))
            if raw in sna or abs(v - rest_value) < 1e-9:
                continue
            out.append(Probe(s.name, round(v, 6), f"{v:g}", {}))
        return out

    def steps(self) -> list[dict[str, Any]]:
        """Rest, then each signal's states, each signal followed by rest again."""
        steps = [{"id": "rest", "kind": "rest", "name": "Rest", "set": {}}]
        for i, sp in enumerate(self.signals):
            for j, p in enumerate(sp.probes):
                steps.append({"id": f"p{i}.{j}", "kind": "probe", "signal": sp.signal,
                              "index": i, "name": f"{sp.name}: {p.label}", "set": p.set})
            steps.append({"id": f"r{i}", "kind": "rest", "name": "Rest", "set": {},
                          "index": i})
        return steps

    # -- what it showed --------------------------------------------------------
    #
    # Each signal is looked at as soon as its states and the rest after it are
    # shot (:meth:`look`), so a run's learning is mostly done by the time its
    # last state is; :meth:`learn` puts the signals together.

    def begin(self, rest_shots: list[np.ndarray], bulb: list[np.ndarray] | None = None) -> None:
        """The rest screen and the bulb check, before any signal is looked at."""
        self._baseline = L._median(rest_shots)
        self._floors = L.noise_floor(rest_shots) if len(rest_shots) > 1 else (0.0, 0.0)
        self._level = L._threshold(self._floors[0], L.LIT_MIN)
        self._change_level = L._threshold(self._floors[1], L.CHANGE_MIN)
        self._slots = bulb_slots(self._baseline, bulb, self._level) if bulb else []
        self._before = self._baseline
        self._seen: dict[int, dict[str, Any]] = {}
        self._drift = np.zeros(self._baseline.shape[:2], bool)

    def look(self, i: int, captured) -> None:
        """Signal ``i``'s states, against the rest before and after them."""
        sp = self.signals[i]
        level, change_level = self._level, self._change_level
        before = self._before
        after_shots = captured.get(f"r{i}")
        after = L._median(after_shots) if after_shots else before
        drift = (L.change(before, after) >= change_level).astype(np.uint8)
        drift = cv2.dilate(drift, np.ones((9, 9), np.uint8)).astype(bool)
        # Empty at rest: where nothing at all is drawn, a lamp's box and round it.
        empty = cv2.erode((before.max(axis=2) < DARK).astype(np.uint8),
                          np.ones((7, 7), np.uint8)).astype(bool)
        states, found, anything = [], [], False
        for j, p in enumerate(sp.probes):
            shots = captured.get(f"p{i}.{j}")
            if not shots:
                continue
            changed = (L.change(before, L._median(shots)) >= change_level) & ~drift
            changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((2, 2), np.uint8))
            changed = _at_least(changed, MIN_CHANGE_PX)
            deltas = [L.lit_delta(before, s) for s in shots]
            lit = L.blobs(np.max(deltas, axis=0), level)
            # A lamp lights where nothing was: a letter that turns into
            # another (P to D) lights pixels the old one did not use, but
            # inside the box of what was drawn there.
            appeared = [b for b in lit
                        if (empty & b.mask).sum() >= 0.9 * b.area and not (drift & b.mask).any()]
            if changed.any() or appeared:
                anything = True
            if sp.kind == "sweep":
                # A number's symbols are lamps only in a lamp's place.
                appeared = [b for b in appeared
                            if any(L._overlap(b.box, slot) > 0.3 for slot in self._slots)]
            if appeared:
                found.append({"signal": sp, "probe": p,
                              "cands": [L._candidate(b, deltas, shots, level) for b in appeared]})
            states.append(np.packbits(changed))
        self._seen[i] = {"found": found, "states": states, "anything": anything,
                         "drift": bool(drift.any())}
        self._drift |= drift
        self._before = after

    def learn(self, captured, bulb: list[np.ndarray] | None = None
              ) -> tuple[L.Library, dict[str, Any]]:
        """Lamps and elements from what every state showed. See the module's doc.

        ``bulb`` is the bulb check's frames, when the start-up was seen. The
        signals :meth:`look` has not already seen are looked at now.
        """
        if not hasattr(self, "_seen"):
            self.begin(captured["rest"], bulb)
        for i in range(len(self.signals)):
            if i not in self._seen:
                self.look(i, captured)
        baseline = self._baseline
        lit_floor, change_floor = self._floors
        shape = baseline.shape[:2]
        problems: list[str] = []
        no_effect = [self.signals[i].name for i in sorted(self._seen)
                     if not self._seen[i]["anything"]]
        drifting = [self.signals[i].name for i in sorted(self._seen)
                    if self._seen[i]["drift"] and self._seen[i]["anything"]]
        found = [f for i in sorted(self._seen) for f in self._seen[i]["found"]]

        # A lamp's size: from the states that lit one thing and nothing else.
        singles = [f["cands"][0]["box"] for f in found if len(f["cands"]) == 1]
        if not singles:
            singles = [c["box"] for f in found for c in f["cands"]]
        tw = float(np.median([b[2] for b in singles])) if singles else 32.0
        th = float(np.median([b[3] for b in singles])) if singles else 32.0
        lo, hi = LAMP_SIZE_RANGE

        def lamp_sized(c) -> bool:
            w, h = c["box"][2], c["box"][3]
            return lo <= w / tw <= hi and lo <= h / th <= hi and 0.33 <= w / max(h, 1) <= 3.0

        lamps: list[L.Lamp] = []
        lamp_boxes: dict[str, list] = {}
        for f in found:
            cands = [c for c in f["cands"] if lamp_sized(c)]
            if not cands:
                continue
            best = min(cands, key=lambda c: (abs(math.log(c["box"][2] / tw))
                                             + abs(math.log(c["box"][3] / th)), -c["score"]))
            sp, p = f["signal"], f["probe"]
            state = {"set": p.set, "flashing": bool(best["flashed"]), "signal": sp.signal}
            name = sp.name
            if sp.kind == "sweep":
                name = f"{sp.name} {self._end(sp, p)}"
                # Taken past its rest value, away from where it lit, before
                # anything else is shown: a warning with hysteresis lets go
                # only when its value comes back from that side (see _probes).
                release = _release(sp, p)
                if release is not None:
                    state["release"] = {sp.signal: release}
            lamp_boxes.setdefault(sp.signal, []).append(best["box"])
            # One lamp, whatever lights it: ABS On and ABS Flash; every door.
            twin = next((lamp for lamp in lamps if _same_symbol(lamp, best, max(tw, th))), None)
            if twin is not None:
                same_signal = any(st["signal"] == sp.signal for st in twin.states.values())
                label = p.label if same_signal else f"{sp.name}: {p.label}"
                twin.states[label] = state
                if twin.blinks and not state["flashing"]:
                    # A steady state is what the lamp is learnt from.
                    _relearn(twin, best, p.set)
                continue
            lamps.append(L.Lamp(
                name=name, set=p.set, blinks=state["flashing"], box=best["box"],
                centre=best["centre"], template=best["template"], hue=best["hue"],
                colour_ratio=best["ratio"], strength=best["strength"],
                flashed=state["flashing"], offset=best["offset"],
                also=[c["box"] for c in f["cands"] if c is not best],
                states={p.label: state}))
        _name_lamps(lamps, self.signals)

        lib = L.Library(baseline=baseline, lamps=lamps, lit_floor=lit_floor,
                        change_floor=change_floor, lamp_size=(tw, th))
        L._set_zone(lib)
        # Elements: a signal's changes less its lamps.
        named_sweeps = []
        for i in sorted(self._seen):
            sp = self.signals[i]
            boxes = lamp_boxes.get(sp.signal, [])
            states = []
            for packed in self._seen[i]["states"]:
                m = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
                for x, y, w, h in boxes:
                    pad = int(round(0.3 * max(w, h)))
                    m[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = False
                states.append(m)
            regions = _regions(states)
            if regions:
                named_sweeps.append({"name": sp.name, "states": states, "regions": regions})
        summary = {
            "lamps": [lamp.to_dict() for lamp in lamps],
            "lamp_size": [round(tw, 1), round(th, 1)],
            "floors": {"lit": round(lit_floor, 2), "change": round(change_floor, 2)},
            "elements": [{"name": s["name"], "regions": [list(b) for b in s["regions"]]}
                         for s in named_sweeps],
            "no_effect": no_effect,
            "changes_with_time": drifting,
            "signals_tried": len(self.signals),
            "problems": problems,
        }
        if no_effect:
            problems.append(f"{len(no_effect)} signal(s) changed nothing on the screen; some "
                            "need another signal first (rig.json 'when'): "
                            + ", ".join(no_effect[:12]) + ("..." if len(no_effect) > 12 else ""))
        summary["_sweeps"] = named_sweeps
        summary["_drift"] = self._drift
        return lib, summary

    def _end(self, sp: SignalProbes, p: Probe) -> str:
        """"low" or "high": which end of its sweep a number lit a lamp at."""
        values = [float(q.value) for q in sp.probes]
        return "low" if float(p.value) <= (min(values) + max(values)) / 2 else "high"

    # -- checking ----------------------------------------------------------------

    def all_state(self, lib: L.Library) -> dict[str, Any]:
        """Every lamp's steady state at once; where two set one signal, the later wins."""
        state: dict[str, Any] = {}
        for lamp in lib.lamps:
            if not lamp.states:
                continue
            steady = [s for s in lamp.states.values() if not s["flashing"]]
            pick = (steady or list(lamp.states.values()) or [{"set": lamp.set}])[0]
            state.update(pick["set"])
        return state

    def check_steps(self, lib: L.Library) -> list[dict[str, Any]]:
        """Rest, every state of every lamp, all lamps together, then rest.

        A lamp a number lit is followed by the number driven past where it
        lit (a "release" step, not checked), so a warning with hysteresis
        does not stay lit into the next state.
        """
        steps = [{"id": "rest0", "kind": "rest", "name": "Rest", "set": {}}]
        releases: dict[str, Any] = {}
        for k, lamp in enumerate(lib.lamps):
            # A lamp nothing on the bus lights (seen in the bulb check only)
            # cannot be switched on to check.
            for label, st in lamp.states.items():
                steps.append({"id": f"c{k}.{len(steps)}", "kind": "lamp", "name": lamp.name,
                              "lamp": lamp.name, "state": label, "set": st["set"],
                              "flashing": bool(st.get("flashing"))})
                if st.get("release"):
                    releases.update(st["release"])
                    steps.append({"id": f"x{k}.{len(steps)}", "kind": "release",
                                  "name": f"{lamp.name}: letting go", "set": st["release"]})
        if len(lib.lamps) > 1:
            steps.append({"id": "all", "kind": "all", "name": "All lamps",
                          "set": self.all_state(lib)})
            if releases:
                steps.append({"id": "xall", "kind": "release", "name": "All lamps: letting go",
                              "set": releases})
        steps.append({"id": "rest2", "kind": "rest", "name": "Rest again", "set": {}})
        return steps


def bulb_slots(baseline: np.ndarray, bulb: list[np.ndarray], level: float
               ) -> list[tuple[int, int, int, int]]:
    """Every lamp's place in the bulb check: compact symbols lit where rest is dark.

    Needles swept through the bulb check light a long fan, not a symbol.
    """
    if not bulb:
        return []
    delta = np.median(np.stack([L.lit_delta(baseline, b) for b in bulb]), axis=0)
    dark = baseline.max(axis=2) < DARK
    boxes = []
    for b in L.blobs(delta, level):
        x, y, w, h = b.box
        if (dark & b.mask).sum() < 0.6 * b.area or not 0.33 <= w / max(h, 1) <= 3.0:
            continue
        boxes.append(b.box)
    if not boxes:
        return []
    size = float(np.median([max(b[2], b[3]) for b in boxes]))
    return [b for b in boxes if max(b[2], b[3]) <= 2.5 * size and max(b[2], b[3]) >= 0.3 * size]


def _at_least(mask: np.ndarray, px: int) -> np.ndarray:
    """``mask`` less the specks smaller than ``px``: noise, not a change."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= px
    return keep[lab]


def _release(sp: SignalProbes, lit: Probe):
    """The swept value nearest rest on the far side of it from where a lamp lit."""
    if sp.rest is None:
        return None
    v = float(lit.value)
    other = [q.value for q in sp.probes if (float(q.value) - sp.rest) * (v - sp.rest) < 0]
    return min(other, key=lambda x: abs(float(x) - sp.rest)) if other else None


def _same_symbol(lamp: L.Lamp, cand: dict, size: float) -> bool:
    """The same lamp: the same symbol, in the same place and colour."""
    cx, cy = lamp.centre
    if math.hypot(cand["centre"][0] - cx, cand["centre"][1] - cy) > SAME_PLACE * size:
        return False
    if (lamp.hue is None) != (cand["hue"] is None):
        return False                            # white, and coloured
    if lamp.hue is not None and L.hue_distance(lamp.hue, cand["hue"]) > L.HUE_TOLERANCE:
        return False
    a, b = lamp.template, cand["template"]
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = cv2.copyMakeBorder(a, 0, h - a.shape[0] + 4, 0, w - a.shape[1] + 4, cv2.BORDER_CONSTANT, 0)
    score = float(cv2.matchTemplate(pa, b, cv2.TM_CCOEFF_NORMED).max())
    return score >= SAME_LAMP


def _relearn(lamp: L.Lamp, cand: dict, state: dict) -> None:
    lamp.box, lamp.centre, lamp.template = cand["box"], cand["centre"], cand["template"]
    lamp.offset, lamp.hue, lamp.strength = cand["offset"], cand["hue"], cand["strength"]
    lamp.set, lamp.blinks, lamp.flashed = state, False, False


def _name_lamps(lamps: list[L.Lamp], signals: list[SignalProbes]) -> None:
    """A signal lighting one lamp lends it its name; lighting several, its name and the state."""
    by_signal: dict[str, list[L.Lamp]] = {}
    for lamp in lamps:
        sig = next(iter(lamp.states.values()))["signal"]
        by_signal.setdefault(sig, []).append(lamp)
    for sig, group in by_signal.items():
        if len(group) > 1:
            for lamp in group:
                lamp.name = f"{lamp.name}: {next(iter(lamp.states))}"
    seen: dict[str, int] = {}
    for lamp in lamps:
        seen[lamp.name] = seen.get(lamp.name, 0) + 1
    for lamp in lamps:
        if seen[lamp.name] > 1:
            lamp.name = f"{lamp.name} ({next(iter(lamp.states.values()))['signal']})"


def _regions(states: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    if not states:
        return []
    union = np.any(states, axis=0)
    if not union.any():
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)))
    return [tuple(int(v) for v in stats[k, :4]) for k in range(1, n)
            if stats[k, cv2.CC_STAT_AREA] >= 30]

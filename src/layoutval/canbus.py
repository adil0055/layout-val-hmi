"""Driving a cluster over CAN, from its DBC alone: the tester's side of the bus.

A production cluster expects what the car gives it: every message it listens
to, at its own cycle time, with alive counters and CRCs where the matrix has
them, the ignition on, and plausible values everywhere else. Miss a message
and it shows the failsafe for it (the ABS lamp for a lost ESC message); send
a bad CRC and the frame is thrown away. So the tester here works the matrix
out from the DBC, the way a rest-bus simulation is configured:

* **which node is the cluster** -- the receiver of most signals, a name like
  IPC, IC, KOMBI or CLUSTER preferred -- and so which messages to send: every
  one the cluster does not send itself;
* **cycle times** from ``GenMsgCycleTime``;
* **alive counters and CRCs** by their names (``_Counter``, ``_AliveCounter``,
  ``_BZ``, ``_CRC``, ``_Checksum``, ...), computed AUTOSAR E2E profile 1 style
  by default; the algorithm and Data ID differ between makers, so they can be
  set (:class:`Rig`);
* **the ignition**: a signal named like KL15, Ignition or PowerMode, with its
  On and Off values from its value table;
* **the rest state**: the start values, except where one means "signal not
  available" or "init", which a cluster shows as a fault: then the value
  named like Off, Closed, Buckled or None.

What was worked out can be written out (:meth:`Rig.to_dict`) and read back
with corrections, as ``rig.json``.

The bus is anything python-can opens, named ``INTERFACE[:CHANNEL][@BITRATE]``:
``udp_multicast`` (a virtual bus over the network, no hardware -- the
default), ``socketcan:vcan0``, ``pcan:PCAN_USBBUS1@500000``, ``kvaser:0@500000``.

It implements :class:`layoutval.teachin.SignalDriver`.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: python-can's virtual bus over UDP multicast, and the group it uses for IPv4.
UDP_GROUP = "239.74.163.2"

CLUSTER_NAMES = re.compile(r"^(ipc|ic|icu|icm|ipk|kombi|cluster|clu|dash|instr\w*|ip|cl)(\b|_|\d)",
                           re.I)
COUNTER = re.compile(r"(alive_?)?(counter|cntr|ctr|cnt|zaehler|bz|sqc|seq(uence)?_?c(ou)?nt(er)?)$",
                     re.I)
CRC = re.compile(r"(crc|chks(u)?m?|checksum|chk|cs)$", re.I)
# The name must start a word: "Nachtdesign" ends in "ign" too.
IGNITION = re.compile(r"(^|_)(kl_?15|klemme_?15|terminal_?15|ign(ition)?(_?(sw|sts|stat|state|status|mode|pos))?"
                      r"|key_?(sts|pos|state)|power_?mode|pwr_?mode|sys_?pwr(_?(mode|sts))?"
                      r"|vehicle_?power(_?mode)?)$", re.I)
#: Network management and its signals ("NM active because of KL15") are not the ignition.
NETWORK = re.compile(r"(^|_)(nm|nmh|osek_?nm|autosar_?nm|wakeup)(_|$)", re.I)
IGN_ON = re.compile(r"^(on|kl_?15_?on|ign(ition)?_?on|run(ning)?|active|ign)$", re.I)
IGN_OFF = re.compile(r"^(off|kl_?15_?off|ign(ition)?_?off|lock(ed)?|inactive)$", re.I)
NOT_AVAILABLE = re.compile(r"(sna|not_?avail|n/?a$|invalid|init|undefined|reserved|error|unknown)",
                           re.I)
QUIET = re.compile(r"^(off|inactive|not_?active|none|no_\w+|normal|ok|closed|close|"
                   r"buckled|fastened|released|idle|stopped|standby|disabled|false|0|"
                   r"no_?warn(ing)?|not_?requested|not_?pressed|detected|present|paused)$", re.I)


def parse_bus(spec: str) -> dict:
    """``INTERFACE[:CHANNEL][@BITRATE]`` as keyword arguments for ``can.Bus``.

    ``udp_multicast`` alone means the IPv4 group, which crosses an ordinary
    network where python-can's IPv6 default may not.
    """
    spec = (spec or "").strip()
    bitrate = None
    if "@" in spec:
        spec, rate = spec.rsplit("@", 1)
        bitrate = int(rate)
    interface, _, channel = spec.partition(":")
    interface = interface.strip() or "udp_multicast"
    if not channel and interface == "udp_multicast":
        channel = UDP_GROUP
    kwargs: dict = {"interface": interface}
    if channel:
        kwargs["channel"] = channel
    if bitrate:
        kwargs["bitrate"] = bitrate
    return kwargs


class CanError(RuntimeError):
    """A signal the DBC does not have, or a value it cannot carry."""


# -- CRCs ----------------------------------------------------------------------

def crc8(data: bytes, poly: int, init: int, xorout: int) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ xorout


ALGORITHMS = {
    # AUTOSAR Crc_CalculateCRC8 (SAE J1850), as E2E profile 1 uses it.
    "crc8_sae_j1850": lambda d: crc8(d, 0x1D, 0xFF, 0xFF),
    # AUTOSAR Crc_CalculateCRC8H2F.
    "crc8_h2f": lambda d: crc8(d, 0x2F, 0xFF, 0xFF),
    "xor": lambda d: _xor(d),
    "sum": lambda d: sum(d) & 0xFF,
}


def _xor(data: bytes) -> int:
    out = 0
    for b in data:
        out ^= b
    return out


@dataclass
class E2E:
    """How one message's counter and CRC are made."""

    counter: str | None = None
    crc: str | None = None
    algorithm: str = "crc8_sae_j1850"
    data_id: int | str = "frame_id"
    """The Data ID mixed into the CRC: a number, or the message's frame ID."""
    data_id_bytes: str = "both"
    """"both" (low byte, then high), "low", or "none"."""
    counter_max: int | None = None
    """Highest counter value before it wraps; default 14 for a 4-bit counter
    with a CRC (profile 1 keeps 15 back), else the counter's full range."""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Rig:
    """Everything the tester needs to know about one cluster's bus."""

    dbc: Path
    cluster_node: str | None = None
    send: list[str] = field(default_factory=list)
    ignition: dict | None = None
    """{"signal": name, "on": value, "off": value}"""
    e2e: dict[str, E2E] = field(default_factory=dict)
    rest: dict[str, Any] = field(default_factory=dict)
    """Rest values that differ from the DBC's start values."""
    exclude: list[str] = field(default_factory=list)
    """Signals or messages never switched on to see what they do."""
    names: dict[str, str] = field(default_factory=dict)
    """Signal (or ``signal=value``) -> what to call what it shows."""
    when: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Signal -> other signals to set while it is tried: the gear engaged
    shows only in D, so {"TCU_GearActual": {"TCU_GearSel": "D"}}."""
    notes: list[str] = field(default_factory=list)

    # -- working it out ------------------------------------------------------------

    @classmethod
    def detect(cls, dbc: str | Path, overrides: dict | None = None) -> "Rig":
        """The rig as far as the DBC tells it, with ``overrides`` (a rig.json) on top."""
        import cantools

        db = cantools.database.load_file(str(dbc))
        o = dict(overrides or {})
        rig = cls(dbc=Path(dbc))
        rig.cluster_node = o.get("cluster_node") or _cluster_node(db, rig.notes)
        node = rig.cluster_node
        rig.send = o.get("send") or [
            m.name for m in db.messages
            if node not in (m.senders or []) and not _diagnostic(m)]
        default_e2e = dict(o.get("e2e_default", {}))
        given = o.get("e2e", {})
        for m in db.messages:
            if m.name not in rig.send:
                continue
            if m.name in given:
                if given[m.name]:
                    rig.e2e[m.name] = E2E(**{**default_e2e, **given[m.name]})
                continue
            counter = next((s.name for s in m.signals if COUNTER.search(s.name)), None)
            crc = next((s.name for s in m.signals if CRC.search(s.name) and s.length == 8), None)
            if counter or crc:
                rig.e2e[m.name] = E2E(counter=counter, crc=crc, **default_e2e)
        rig.ignition = o.get("ignition") or _ignition(db, rig.send, rig.notes)
        rig.exclude = list(o.get("exclude", []))
        rig.names = dict(o.get("names", {}))
        rig.when = {k: dict(v) for k, v in o.get("when", {}).items()}
        rest = _rest(db, rig)
        rest.update(o.get("rest", {}))
        rig.rest = rest
        return rig

    @classmethod
    def load(cls, path: str | Path, dbc: str | Path | None = None) -> "Rig":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        dbc = dbc or raw.get("dbc")
        if not dbc:
            raise ValueError(f"{path.name} names no DBC; pass --dbc")
        dbc = Path(dbc)
        if not dbc.is_absolute():
            dbc = path.parent / dbc
        return cls.detect(dbc, raw)

    def to_dict(self) -> dict:
        return {
            "dbc": str(self.dbc),
            "cluster_node": self.cluster_node,
            "send": self.send,
            "ignition": self.ignition,
            "e2e": {k: v.to_dict() for k, v in self.e2e.items()},
            "rest": self.rest,
            "exclude": self.exclude,
            "names": self.names,
            "when": self.when,
            "notes": self.notes,
        }


def _diagnostic(m) -> bool:
    name = m.name.lower()
    return (name.startswith(("diag", "uds", "xcp", "obd", "ccp")) or "_diag" in name
            or 0x700 <= m.frame_id <= 0x7FF and not m.is_extended_frame)


def _cluster_node(db, notes: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for m in db.messages:
        for s in m.signals:
            for r in s.receivers or []:
                counts[r] = counts.get(r, 0) + 1
    if not counts:
        notes.append("no receivers in the DBC: every message is sent")
        return None
    named = [n for n in counts if CLUSTER_NAMES.search(n)]
    pick = max(named or counts, key=lambda n: counts[n])
    notes.append(f"cluster node: {pick} (receives {counts[pick]} signals)"
                 + ("" if named else "; no node is named like a cluster, so the busiest receiver"))
    return pick


def _ignition(db, send: list[str], notes: list[str]) -> dict | None:
    found = []
    for m in db.messages:
        if m.name not in send:
            continue
        if NETWORK.search(m.name):
            continue
        for s in m.signals:
            if not IGNITION.search(s.name) or NETWORK.search(s.name):
                continue
            choices = {int(k): str(v) for k, v in (s.choices or {}).items()}
            on = next((v for v in choices.values() if IGN_ON.match(v)), None)
            off = next((v for v in choices.values() if IGN_OFF.match(v)), None)
            if not choices and s.length == 1:
                on, off = 1, 0
            if on is not None and off is not None:
                # A value table saying On and Off outranks a bare bit.
                found.append((0 if choices else 1, s.name, on, off))
    if found:
        _, name, on, off = min(found)
        notes.append(f"ignition: {name} ({off} / {on})")
        return {"signal": name, "on": on, "off": off}
    notes.append("no ignition signal found: the cluster must already be awake")
    return None


def _rest(db, rig: Rig) -> dict:
    """Rest values that differ from the start values: the cluster awake, nothing lit."""
    rest: dict[str, Any] = {}
    e2e_signals = {n for e in rig.e2e.values() for n in (e.counter, e.crc) if n}
    for m in db.messages:
        if m.name not in rig.send:
            continue
        for s in m.signals:
            if s.name in e2e_signals:
                continue
            start = int(s.raw_initial or 0)
            choices = {int(k): str(v) for k, v in (s.choices or {}).items()}
            name = choices.get(start)
            if name is not None and NOT_AVAILABLE.search(name):
                quiet = [v for v in choices.values()
                         if QUIET.match(v) and not NOT_AVAILABLE.search(v)]
                if quiet:
                    rest[s.name] = quiet[0]
                    rig.notes.append(f"rest: {s.name} = {quiet[0]} (its start value is {name})")
            elif name is None and start == (1 << s.length) - 1 and s.length >= 4:
                rest[s.name] = s.minimum if s.minimum is not None else 0
    if rig.ignition:
        rest[rig.ignition["signal"]] = rig.ignition["on"]
    return rest


# -- the tester ------------------------------------------------------------------

class CanDriver:
    """Every message the cluster listens to, held and sent at its cycle time.

    ``source`` is a DBC path or a :class:`Rig`.
    """

    def __init__(self, source: str | Path | Rig, bus: str = "udp_multicast", *,
                 period_s: float = 0.1, bus_obj: Any = None) -> None:
        import cantools

        self.rig = source if isinstance(source, Rig) else Rig.detect(source)
        self.dbc_path = self.rig.dbc
        self.db = cantools.database.load_file(str(self.rig.dbc))
        self.messages = [m for m in self.db.messages if m.name in set(self.rig.send)]
        self._by_signal = {s.name: m for m in self.messages for s in m.signals}
        self._period = {m.name: (m.cycle_time or period_s * 1000.0) / 1000.0 for m in self.messages}
        self._e2e = {name: e for name, e in self.rig.e2e.items()}
        self._counter = {m.name: 0 for m in self.messages}
        start = {m.name: {s.name: int(s.raw_initial or 0) for s in m.signals} for m in self.messages}
        for name, value in self.rig.rest.items():
            if name in self._by_signal:
                start[self._by_signal[name].name][name] = self._raw(name, value)
        self._rest = start
        self._state = {k: dict(v) for k, v in self._rest.items()}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._quiet = False
        if bus_obj is None:
            import can
            self.bus_spec = parse_bus(bus)
            bus_obj = can.Bus(**self.bus_spec)
        else:
            self.bus_spec = {"interface": "given"}
        self.bus = bus_obj
        self.sent = 0
        self._due = {m.name: 0.0 for m in self.messages}
        self._thread = threading.Thread(target=self._cycle, name="can-tester", daemon=True)
        self._thread.start()

    # -- the SignalDriver protocol ---------------------------------------------

    def set(self, signal: str, value: Any) -> None:
        self.apply({signal: value}, from_rest=False)

    # -- states ----------------------------------------------------------------

    @property
    def signals(self) -> list[str]:
        return list(self._by_signal)

    def check(self, values: dict[str, Any]) -> None:
        """Raise CanError for a signal the bus does not carry or a value it cannot."""
        for name, value in values.items():
            self._raw(name, value)

    def apply(self, values: dict[str, Any], *, from_rest: bool = True) -> None:
        """Put these signals on the bus, everything else at rest unless ``from_rest`` is off.

        The messages that changed are sent at once; every message keeps
        going out on its cycle.
        """
        raws = {name: self._raw(name, value) for name, value in values.items()}
        with self._lock:
            before = {k: dict(v) for k, v in self._state.items()}
            if from_rest:
                self._state = {k: dict(v) for k, v in self._rest.items()}
            for name, raw in raws.items():
                self._state[self._by_signal[name].name][name] = raw
            if not self._quiet:
                for m in self.messages:
                    if self._state[m.name] != before[m.name]:
                        self._send(m)

    def rest(self) -> None:
        self.apply({})

    def ignition(self, on: bool) -> bool:
        """Switch the ignition, everything else at rest. False if there is none."""
        ign = self.rig.ignition
        if not ign:
            return False
        self.apply({ign["signal"]: ign["on"] if on else ign["off"]})
        return True

    def quiet(self, silent: bool) -> None:
        """Stop sending altogether (the bus asleep), or start again."""
        self._quiet = silent

    def value(self, signal: str) -> float:
        """A signal's current value, scaled."""
        m = self._by_signal[signal]
        with self._lock:
            raw = self._state[m.name][signal]
        return float(m.get_signal_by_name(signal).raw_to_scaled(raw))

    def raw_state(self, message: str) -> dict:
        with self._lock:
            return dict(self._state[message])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.bus.shutdown()
        except Exception:
            pass

    # -- internals -------------------------------------------------------------

    def _raw(self, name: str, value: Any) -> int:
        m = self._by_signal.get(name)
        if m is None:
            raise CanError(f"{name!r} is not a signal the tester sends "
                           f"({Path(self.dbc_path).name})")
        sig = m.get_signal_by_name(name)
        if isinstance(value, str):
            try:
                return int(sig.choice_to_number(value))
            except (KeyError, AttributeError, TypeError):
                choices = ", ".join(str(c) for c in (sig.choices or {}).values())
                raise CanError(f"{name} has no value {value!r}"
                               + (f"; it takes {choices}" if choices else "")) from None
        number = float(value)
        if (sig.minimum is not None and number < sig.minimum - 1e-9) or \
                (sig.maximum is not None and number > sig.maximum + 1e-9):
            raise CanError(f"{name} = {value} is outside {sig.minimum}..{sig.maximum}")
        return int(round(float(sig.scaled_to_raw(number))))

    def _encode(self, m) -> bytes:
        raw = dict(self._state[m.name])
        e = self._e2e.get(m.name)
        if e is None:
            return m.encode(raw, scaling=False, strict=False)
        if e.counter:
            sig = m.get_signal_by_name(e.counter)
            # Profile 1 keeps a 4-bit counter's 15 back; it goes with a CRC.
            top = e.counter_max if e.counter_max is not None else (
                14 if sig.length == 4 and e.crc else (1 << sig.length) - 1)
            raw[e.counter] = self._counter[m.name] % (top + 1)
            self._counter[m.name] = (self._counter[m.name] + 1) % (top + 1)
        if not e.crc:
            return m.encode(raw, scaling=False, strict=False)
        crc_sig = m.get_signal_by_name(e.crc)
        raw[e.crc] = 0
        data = bytearray(m.encode(raw, scaling=False, strict=False))
        at = crc_sig.start // 8
        data_id = m.frame_id if e.data_id == "frame_id" else int(e.data_id)
        prefix = {"both": bytes([data_id & 0xFF, (data_id >> 8) & 0xFF]),
                  "low": bytes([data_id & 0xFF]), "none": b""}[e.data_id_bytes]
        body = bytes(b for i, b in enumerate(data) if i != at)
        data[at] = ALGORITHMS[e.algorithm](prefix + body)
        return bytes(data)

    def _send(self, m) -> None:
        import can
        self.bus.send(can.Message(arbitration_id=m.frame_id, data=self._encode(m),
                                  is_extended_id=m.is_extended_frame))
        self.sent += 1
        self._due[m.name] = time.monotonic() + self._period[m.name]

    def _cycle(self) -> None:
        while not self._stop.wait(0.005):
            if self._quiet:
                continue
            now = time.monotonic()
            with self._lock:
                try:
                    for m in self.messages:
                        if now >= self._due[m.name]:
                            self._send(m)
                except Exception:
                    # A bus that went away: the next apply() says so to its caller.
                    time.sleep(0.05)

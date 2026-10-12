"""Driving the cluster over CAN: the tester's side of the bus.

A :class:`CanDriver` holds the value of every signal in a DBC and sends every
message cyclically, as a vehicle's network does: a cluster that joins the bus
late, or misses a frame, still converges on the state. Changing a signal sends
its message at once rather than waiting for the cycle, so a capture can follow
quickly.

The bus is anything python-can opens, named ``INTERFACE[:CHANNEL][@BITRATE]``:
``udp_multicast`` (a virtual bus over the network, no hardware -- the default),
``socketcan:vcan0``, ``pcan:PCAN_USBBUS1@500000``, ``kvaser:0@500000``, ...

It implements :class:`layoutval.teachin.SignalDriver`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

#: python-can's virtual bus over UDP multicast, and the group it uses for IPv4.
UDP_GROUP = "239.74.163.2"


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


class CanDriver:
    """Every signal of a DBC, held and sent cyclically on one bus."""

    def __init__(self, dbc: str | Path, bus: str = "udp_multicast", *,
                 period_s: float = 0.1, bus_obj: Any = None) -> None:
        import cantools

        self.db = cantools.database.load_file(str(dbc))
        self.dbc_path = Path(dbc)
        self.period_s = period_s
        self._by_signal = {s.name: m for m in self.db.messages for s in m.signals}
        self._rest = {m.name: {s.name: int(s.raw_initial or 0) for s in m.signals}
                      for m in self.db.messages}
        self._state = {k: dict(v) for k, v in self._rest.items()}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if bus_obj is None:
            import can
            self.bus_spec = parse_bus(bus)
            bus_obj = can.Bus(**self.bus_spec)
        else:
            self.bus_spec = {"interface": "given"}
        self.bus = bus_obj
        self.sent = 0
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
        """Raise CanError for a signal the DBC lacks or a value it cannot carry."""
        for name, value in values.items():
            self._raw(name, value)

    def apply(self, values: dict[str, Any], *, from_rest: bool = True) -> None:
        """Put these signals on the bus, everything else at rest unless ``from_rest`` is off.

        The messages that changed are sent at once; every message keeps
        going out on the cycle.
        """
        raws = {name: self._raw(name, value) for name, value in values.items()}
        with self._lock:
            before = {k: dict(v) for k, v in self._state.items()}
            if from_rest:
                self._state = {k: dict(v) for k, v in self._rest.items()}
            for name, raw in raws.items():
                self._state[self._by_signal[name].name][name] = raw
            changed = [m for m in self.db.messages if self._state[m.name] != before[m.name]]
            for m in changed:
                self._send(m)

    def rest(self) -> None:
        self.apply({})

    def value(self, signal: str) -> float:
        """A signal's current value, scaled."""
        m = self._by_signal[signal]
        with self._lock:
            raw = self._state[m.name][signal]
        return float(m.get_signal_by_name(signal).raw_to_scaled(raw))

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
            raise CanError(f"{name!r} is not a signal in {self.dbc_path.name}")
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

    def _send(self, m) -> None:
        import can
        data = m.encode(self._state[m.name], scaling=False, strict=False)
        self.bus.send(can.Message(arbitration_id=m.frame_id, data=data,
                                  is_extended_id=m.is_extended_frame))
        self.sent += 1

    def _cycle(self) -> None:
        while not self._stop.wait(self.period_s):
            with self._lock:
                try:
                    for m in self.db.messages:
                        self._send(m)
                except Exception:
                    # A bus that went away: the next apply() says so to its caller.
                    pass

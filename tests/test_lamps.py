"""The CAN teach-in and the lamp check, on a small cluster of its own.

The cluster here draws its lamps the way the HMI does: a strip in which only
lit lamps take space, centred, so every lamp lit alone lights in the same
middle slot, and a flashing lamp keeps its slot through the flash. Each state
is shot through the simulated camera and rectified back, as the server does.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from layoutval import lamps as L
from layoutval.calibration import Calibration, DisplayGeometry
from layoutval.simulator import VirtualCamera

DS = (640, 260)
SIZE, GAP, ROW_Y = 30, 12, 46

RED, AMBER, GREEN, BLUE, WHITE = (40, 50, 235), (20, 160, 245), (90, 200, 70), (240, 140, 40), (235, 235, 235)


def _brake(g, c):
    cv2.circle(g, (15, 15), 11, c, 2, cv2.LINE_AA)
    cv2.line(g, (15, 9), (15, 17), c, 3)
    cv2.circle(g, (15, 21), 2, c, -1)


def _battery(g, c):
    cv2.rectangle(g, (4, 9), (26, 24), c, 2)
    cv2.rectangle(g, (8, 5), (11, 9), c, -1)
    cv2.rectangle(g, (19, 5), (22, 9), c, -1)


def _abs(g, c):
    cv2.circle(g, (15, 15), 12, c, 2, cv2.LINE_AA)
    cv2.putText(g, "A", (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


def _belt(g, c):
    cv2.circle(g, (15, 6), 4, c, -1)
    cv2.line(g, (15, 10), (15, 22), c, 4)
    cv2.line(g, (8, 12), (22, 24), c, 2)


def _beam(g, c):
    cv2.ellipse(g, (12, 15), (8, 11), 0, -90, 90, c, -1)
    for y in (7, 15, 23):
        cv2.line(g, (20, y), (28, y), c, 2)


def _cruise(g, c):
    cv2.ellipse(g, (15, 18), (11, 11), 0, 180, 360, c, 2, cv2.LINE_AA)
    cv2.line(g, (15, 18), (21, 10), c, 2)


LAMPS = [  # name, signal, drawing, colour, flashes
    ("Brake", "brake", _brake, RED, False),
    ("Battery", "battery", _battery, RED, False),
    ("Seat belt", "belt", _belt, RED, True),
    ("ABS", "abs", _abs, AMBER, False),
    ("High beam", "beam", _beam, BLUE, False),
    ("Cruise", "cruise", _cruise, GREEN, False),
    ("Cruise ready", "ready", _cruise, WHITE, False),
]


def render(state: dict, *, blink_on: bool | float = True, faults: dict | None = None) -> np.ndarray:
    """The cluster in ``state``; ``faults`` draws it wrong on purpose."""
    faults = faults or {}
    img = np.full((DS[1], DS[0], 3), (14, 12, 10), np.uint8)
    cv2.circle(img, (160, 170), 70, (70, 60, 50), 2, cv2.LINE_AA)
    cv2.putText(img, f"{state.get('speed', 0):.0f}", (130, 185), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, "km/h", (138, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(img, "TRIP", (430, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    lit = []
    for name, sig, draw, colour, flashes in LAMPS:
        on = bool(state.get(sig))
        if sig == "ready":
            on = bool(state.get("ready")) and not state.get("cruise")
        if on:
            lit.append((name, draw, colour, flashes))
    order = faults.get("order")
    if order:
        lit.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    width = len(lit) * SIZE + (len(lit) - 1) * GAP
    x0 = DS[0] // 2 - width // 2
    for k, (name, draw, colour, flashes) in enumerate(lit):
        if flashes and not blink_on:
            continue                         # its slot kept, dark
        if flashes and blink_on is not True:
            colour = tuple(int(c * blink_on) for c in colour)   # caught mid-switch
        if name in faults.get("missing", ()):
            continue
        colour = faults.get("colour", {}).get(name, colour)
        tile = np.zeros((SIZE, SIZE, 3), np.uint8)
        draw(tile, colour)
        dx, dy = faults.get("shift", {}).get(name, (0, 0))
        x, y = x0 + k * (SIZE + GAP) + dx, ROW_Y - SIZE // 2 + dy
        img[y:y + SIZE, x:x + SIZE] = np.maximum(img[y:y + SIZE, x:x + SIZE], tile)
    if faults.get("stranger"):
        cv2.drawMarker(img, (DS[0] // 2, ROW_Y), AMBER, cv2.MARKER_TILTED_CROSS, 24, 3)
    return img


PLAN = L.TeachPlan(
    lamps=[L.LampStep(name, {sig: 1}, flashes) for name, sig, _, _, flashes in LAMPS],
    elements=[L.SweepStep("Speed", [{"speed": 48}, {"speed": 117}, {"speed": 206}])],
)


@pytest.fixture(scope="module")
def camera():
    return VirtualCamera(display_size=DS, sensor_size=(840, 400), sampling_ratio=1.2,
                         tilt_deg=2.0, roll_deg=0.6, k1=0.0, k2=0.0, defocus_sigma=0.8,
                         noise_sigma=1.2, pwm_amplitude=0.06, seed=4)


def shoot(cam, img):
    return DisplayGeometry(H=cam.H_true, display_size=DS).rectify(cam.shoot(img))


def shots_of(cam, step, **kw):
    """A step's shots; a flashing lamp is caught in both halves of its flash."""
    return [shoot(cam, render(step["set"], blink_on=(k % 3 != 2), **kw))
            for k in range(step["shots"])]


@pytest.fixture(scope="module")
def taught(camera):
    captured = {s["id"]: shots_of(camera, s) for s in PLAN.steps()}
    return L.learn(PLAN, captured)


def test_the_plan_shoots_a_flashing_lamp_across_its_flash():
    steps = {s["id"]: s for s in PLAN.steps()}
    assert steps["lamp2"]["shots"] == 6 and steps["lamp0"]["shots"] == 2
    assert steps["rest"]["shots"] >= 3 and "all" in steps
    assert [s["id"] for s in PLAN.steps() if s["kind"] == "sweep"] == ["el0.0", "el0.1", "el0.2"]
    assert PLAN.all_lamps() == {"brake": 1, "battery": 1, "belt": 1, "abs": 1, "beam": 1,
                                "cruise": 1, "ready": 1}


def test_a_plan_file_with_a_name_used_twice_is_refused(tmp_path):
    f = tmp_path / "plan.json"
    f.write_text(json.dumps({"lamps": [{"name": "ABS", "set": {"a": 1}},
                                       {"name": "ABS", "set": {"b": 1}}]}))
    with pytest.raises(ValueError, match="ABS"):
        L.TeachPlan.load(f)


def test_every_lamp_is_learnt_by_its_symbol_and_colour(taught):
    lib, summary = taught
    assert [lamp.name for lamp in lib.lamps] == [n for n, *_ in LAMPS]
    colours = {lamp.name: L.hue_name(lamp.hue) for lamp in lib.lamps}
    assert colours == {"Brake": "red", "Battery": "red", "Seat belt": "red", "ABS": "amber",
                       "High beam": "blue", "Cruise": "green", "Cruise ready": "white"}
    belt = lib.get("Seat belt")
    assert belt.flashed and not lib.get("Brake").flashed
    # Lit alone, every lamp is in the strip's middle slot.
    slot = DS[0] // 2 - SIZE // 2
    for lamp in lib.lamps:
        x, y, w, h = lamp.box
        assert slot - 2 <= x and x + w <= slot + SIZE + 2, lamp.name


def test_lit_together_each_has_its_slot_in_order(taught):
    lib, summary = taught
    slots = [(lamp.order_all, lamp.name) for lamp in lib.lamps if lamp.slot_all is not None]
    # Cruise ready cannot be lit with Cruise set; the rest can.
    assert [n for _, n in sorted(slots)] == ["Brake", "Battery", "Seat belt", "ABS",
                                             "High beam", "Cruise"]
    assert lib.pitch == pytest.approx(SIZE + GAP, abs=1.0)


def test_a_readout_is_named_by_the_sweep_that_changes_it(taught):
    lib, summary = taught
    sweeps = summary["_sweeps"]
    assert sweeps[0]["regions"], "the speed sweep changed nothing"

    class Spec:
        def __init__(self, id, bbox):
            self.id, self.bbox = id, bbox
    profile = [Spec("digits", (128, 150, 60, 40)), Spec("unit", (136, 203, 40, 16)),
               Spec("trip", (428, 145, 50, 20))]
    names = L.name_elements(profile, sweeps)
    assert names == {"digits": "Speed", "unit": "unknown", "trip": "unknown"}
    mapping = L.rename(profile, names)
    assert [p.id for p in profile] == ["Speed", "unknown@136,203", "unknown@428,145"]
    assert mapping["digits"] == "Speed"


def check(lib, cam, state, expected, where="alone", **faults):
    steps = [{"set": state, "shots": 6}]
    found = lib.find(shots_of(cam, steps[0], faults=faults))
    return L.check(lib, found, expected, where=where)


@pytest.mark.parametrize("name,sig", [(n, s) for n, s, *_ in LAMPS])
def test_each_lamp_alone_passes_where_it_was_taught(taught, camera, name, sig):
    lib, _ = taught
    results = check(lib, camera, {sig: 1}, [name])
    assert L.verdict(results) == "PASS", [r.to_dict() for r in results]


def test_all_lit_together_passes(taught, camera):
    lib, _ = taught
    lit = [lamp.name for lamp in lib.lamps if lamp.slot_all is not None]
    results = check(lib, camera, PLAN.all_lamps(), lit, where="all")
    assert L.verdict(results) == "PASS", [r.to_dict() for r in results if r.verdict != "PASS"]


def _reasons(results):
    return {(r.name, r.reason) for r in results if r.verdict != "PASS"}


def test_a_lamp_that_does_not_light_is_missing(taught, camera):
    lib, _ = taught
    assert _reasons(check(lib, camera, {"abs": 1}, ["ABS"], missing=["ABS"])) == {("ABS", "missing")}


def test_another_lamp_in_its_place_is_both_missing_and_unexpected(taught, camera):
    lib, _ = taught
    assert _reasons(check(lib, camera, {"battery": 1}, ["ABS"])) == {
        ("ABS", "missing"), ("Battery", "unexpected")}


def test_the_right_symbol_in_the_wrong_colour_says_so(taught, camera):
    lib, _ = taught
    results = check(lib, camera, {"abs": 1}, ["ABS"], colour={"ABS": GREEN})
    assert _reasons(results) == {("ABS", "wrong_colour")}


def test_the_same_symbol_in_two_colours_is_two_lamps(taught, camera):
    lib, _ = taught
    assert L.verdict(check(lib, camera, {"ready": 1}, ["Cruise ready"])) == "PASS"
    assert _reasons(check(lib, camera, {"ready": 1}, ["Cruise"])) == {
        ("Cruise", "missing"), ("Cruise ready", "unexpected")}


def test_a_lamp_three_pixels_out_fails_and_one_pixel_passes(taught, camera):
    lib, _ = taught
    far = check(lib, camera, {"abs": 1}, ["ABS"], shift={"ABS": (3, 0)})
    assert _reasons(far) == {("ABS", "position")}
    r = far[0]
    assert math.hypot(r.dx, r.dy) == pytest.approx(3.0, abs=0.6)
    near = check(lib, camera, {"abs": 1}, ["ABS"], shift={"ABS": (1, 0)})
    assert L.verdict(near) in ("PASS", "REVIEW")


def test_a_symbol_no_lamp_has_is_unknown(taught, camera):
    lib, _ = taught
    results = check(lib, camera, {}, [], stranger=True)
    assert [(r.name, r.reason.split(":")[0]) for r in results] == [("unknown", "unknown_symbol")]


def test_a_flashing_lamp_held_steady_is_for_review(taught, camera):
    lib, _ = taught
    steps = {"set": {"belt": 1}, "shots": 6}
    steady = [shoot(camera, render(steps["set"], blink_on=True)) for _ in range(6)]
    results = L.check(lib, lib.find(steady), ["Seat belt"], where="alone")
    assert _reasons(results) == {("Seat belt", "not_flashing")}
    assert L.verdict(results) == "REVIEW"


def test_a_flash_caught_switching_is_still_the_lamp(taught, camera):
    lib, _ = taught
    # A capture is a short mean of frames, and a flash that switches during
    # it leaves the lamp half lit: on the page, through the camera, that was
    # "unknown symbol" for every flashing lamp until half was let in.
    phases = [True, 0.5, False, True, 0.45, False]
    state = {"belt": 1, "brake": 1, "abs": 1}
    shots = [shoot(camera, render(state, blink_on=b)) for b in phases]
    results = L.check(lib, lib.find(shots), ["Brake", "Seat belt", "ABS"])
    assert L.verdict(results) == "PASS", [r.to_dict() for r in results if r.verdict != "PASS"]
    assert next(r for r in results if r.name == "Seat belt").flashing is True


def test_two_lamps_swapped_in_the_strip_fail_and_only_they_do(taught, camera):
    lib, _ = taught
    lit = [lamp.name for lamp in lib.lamps if lamp.slot_all is not None]
    order = ["Battery", "Brake", "Seat belt", "ABS", "High beam", "Cruise"]
    results = check(lib, camera, PLAN.all_lamps(), lit, where="all", order=order)
    assert {r.name for r in results if r.verdict != "PASS"} == {"Brake", "Battery"}


def test_out_of_order_away_from_a_taught_state(taught, camera):
    lib, _ = taught
    # Not a state the lamps were taught in: places are not checked, order is.
    results = check(lib, camera, {"brake": 1, "abs": 1}, ["Brake", "ABS"], where="",
                    order=["ABS", "Brake"])
    assert _reasons(results) == {("Brake", "order")} or _reasons(results) == {("ABS", "order")}
    assert L.verdict(check(lib, camera, {"brake": 1, "abs": 1}, ["Brake", "ABS"], where="")) == "PASS"


def test_a_strip_that_jumps_as_a_lamp_flashes_is_caught(taught, camera):
    lib, _ = taught
    # The bug the HMI had: a flashing lamp's slot vanished in its dark half,
    # and every lamp after it moved over.
    shots = []
    for k in range(6):
        state = {"brake": 1, "abs": 1, "beam": 1}
        if k % 3 != 2:
            state["belt"] = 1
        shots.append(shoot(camera, render(state)))
    results = L.check(lib, lib.find(shots), ["Brake", "Seat belt", "ABS", "High beam"])
    assert ("ABS", "moves_between_shots") in _reasons(results)


def test_the_library_survives_a_save_and_load(taught, camera, tmp_path):
    lib, _ = taught
    lib.save(tmp_path)
    again = L.Library.load(tmp_path)
    assert [lamp.name for lamp in again.lamps] == [lamp.name for lamp in lib.lamps]
    assert L.verdict(check(again, camera, {"abs": 1}, ["ABS"])) == "PASS"


# -- the server ----------------------------------------------------------------

class FakeCan:
    bus_spec = {"interface": "fake"}

    def __init__(self):
        self.applied = []

    def apply(self, values):
        self.applied.append(dict(values))

    def rest(self):
        self.applied.append({})

    def check(self, values):
        pass


def test_the_server_teaches_names_checks_and_validates(camera, tmp_path):
    from layoutval.server import CaptureSession

    can = FakeCan()
    s = CaptureSession(
        tmp_path, display_size=DS, calib_mode="manual", can=can, teach_plan=PLAN,
        calibration=Calibration(intrinsics=None,
                                geometry=DisplayGeometry(H=camera.H_true, display_size=DS,
                                                         method="test")))
    png = lambda img: cv2.imencode(".png", img)[1].tobytes()  # noqa: E731

    def run(mode, faults=None):
        for st in s.can_start(mode)["steps"]:
            s.can_state(st["id"], mode)
            assert can.applied[-1] == st["set"]
            for k in range(st["shots"]):
                img = camera.shoot(render(st["set"], blink_on=(k % 3 != 2), faults=faults))
                rec = s.handle("teach" if mode == "teach" else "lamps", png(img),
                               step=st["id"], shot=k)
                assert rec.verdict != "FAILED", rec.detail
        return s.can_finish(mode)

    with pytest.raises(ValueError, match="teach the lamps first"):
        s.can_start("check")
    rec = run("teach")
    assert rec.verdict == "OK", rec.detail
    assert "7 of 7 lamps" in rec.detail
    assert can.applied[-1] == {}                      # back at rest
    assert s.status()["can"]["taught"] and s.reference is not None
    assert "Speed" in [spec.id for spec in s.profile]
    assert (tmp_path / "teach" / "lamps.json").is_file()
    rec = run("check")
    assert rec.verdict == "PASS", rec.rows
    rec = run("check", faults={"missing": ["ABS"]})
    assert rec.verdict == "FAIL" and {"id": "ABS", "verdict": "FAIL",
                                      "detail": "missing"} in rec.rows
    # A normal validation names a lamp lit when every signal is off.
    v = s.handle("validate", png(camera.shoot(render({}))))
    assert not [f for f in v.report["flags"] if f["flag"] == "lamps_lit"]
    v = s.handle("validate", png(camera.shoot(render({"abs": 1}))))
    lit = [f for f in v.report["flags"] if f["flag"] == "lamps_lit"]
    assert lit and lit[0]["lamps"] == ["ABS"] and v.verdict == "FAIL"


# -- the bus -------------------------------------------------------------------

DBC = """VERSION ""
NS_ :
BS_:
BU_: T C
BO_ 256 LAMPS: 2 T
 SG_ abs : 0|1@1+ (1,0) [0|1] "" C
 SG_ belt : 1|1@1+ (1,0) [0|1] "" C
 SG_ gear : 2|2@1+ (1,0) [0|3] "" C
 SG_ speed : 4|12@1+ (0.1,0) [0|300] "" C
BA_DEF_ SG_ "GenSigStartValue" INT 0 65535;
BA_DEF_DEF_ "GenSigStartValue" 0;
BA_ "GenSigStartValue" SG_ 256 belt 1;
VAL_ 256 gear 0 "P" 1 "R" 2 "N" 3 "D" ;
"""


def test_the_can_driver_sends_states_from_rest(tmp_path):
    can = pytest.importorskip("can")
    pytest.importorskip("cantools")
    from layoutval.canbus import CanDriver, CanError, parse_bus

    assert parse_bus("udp_multicast") == {"interface": "udp_multicast", "channel": "239.74.163.2"}
    assert parse_bus("socketcan:vcan0") == {"interface": "socketcan", "channel": "vcan0"}
    assert parse_bus("pcan:PCAN_USBBUS1@500000")["bitrate"] == 500000
    dbc = tmp_path / "t.dbc"
    dbc.write_text(DBC)
    channel = f"lamps-{id(tmp_path)}"
    listen = can.Bus(interface="virtual", channel=channel)
    drv = CanDriver(dbc, bus_obj=can.Bus(interface="virtual", channel=channel), period_s=10)
    try:
        import cantools
        db = cantools.database.load_file(str(dbc))

        def last():
            msg, got = None, listen.recv(timeout=1.0)
            while got is not None:
                msg, got = got, listen.recv(timeout=0.05)
            return db.decode_message(msg.arbitration_id, msg.data)

        drv.apply({"abs": 1, "gear": "D", "speed": 120.5})
        d = last()
        assert d["abs"] == 1 and str(d["gear"]) == "D" and d["speed"] == pytest.approx(120.5)
        assert d["belt"] == 1                         # its start value: fastened
        drv.apply({"belt": 0})
        d = last()
        assert d["abs"] == 0 and d["belt"] == 0       # from rest: ABS off again
        drv.set("abs", True)                          # SignalDriver: on top of the state
        assert last()["belt"] == 0
        with pytest.raises(CanError, match="not a signal"):
            drv.check({"nope": 1})
        with pytest.raises(CanError, match="no value 'X'"):
            drv.check({"gear": "X"})
        with pytest.raises(CanError, match="outside"):
            drv.check({"speed": 999})
    finally:
        drv.close()
        listen.shutdown()


def test_the_page_runs_a_can_teach_in_over_http(camera, tmp_path):
    import urllib.error
    import urllib.request
    import uuid

    from layoutval.server import PAGE, CaptureSession, serve

    assert 'id="cancard"' in PAGE and "canRun(\"teach\")" in PAGE and "&shot=${k}" in PAGE
    can = FakeCan()
    plan = L.TeachPlan(lamps=PLAN.lamps[:2], elements=[])
    s = CaptureSession(
        tmp_path, display_size=DS, calib_mode="manual", can=can, teach_plan=plan,
        calibration=Calibration(intrinsics=None,
                                geometry=DisplayGeometry(H=camera.H_true, display_size=DS,
                                                         method="test")))
    srv = serve(s, host="127.0.0.1", port=0, quiet=True)
    host, port = srv.server_address
    base = f"http://{host}:{port}"

    def post(path, body):
        req = urllib.request.Request(f"{base}{path}?t={srv.token}", method="POST",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def upload(step, shot, img):
        boundary = uuid.uuid4().hex
        data = cv2.imencode(".png", img)[1].tobytes()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"action\"\r\n\r\nteach\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"s.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + data + \
            f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{base}/upload?t={srv.token}&step={step}&shot={shot}", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    try:
        assert post("/can/start", {"mode": "check"})[0] == 400       # nothing taught yet
        assert post("/can/start", {"mode": "sideways"})[0] == 400
        status, started = post("/can/start", {"mode": "teach"})
        assert status == 200
        for st in started["steps"]:
            status, applied = post("/can/state", {"step": st["id"], "mode": "teach"})
            assert status == 200 and can.applied[-1] == st["set"]
            for k in range(st["shots"]):
                res = upload(st["id"], k, camera.shoot(render(st["set"])))
                assert res["verdict"] == "OK", res
        assert post("/can/state", {"step": "nope", "mode": "teach"})[0] == 400
        status, done = post("/can/finish", {"mode": "teach"})
        assert status == 200 and done["verdict"] == "OK", done
        assert [r["id"] for r in done["rows"]][:2] == ["Brake", "Battery"]
        assert done["overlay"].endswith("-lamps.png")
        assert post("/can/finish", {"mode": "abort"})[0] == 200
        assert can.applied[-1] == {}
        with urllib.request.urlopen(f"{base}/status?t={srv.token}", timeout=10) as r:
            assert json.loads(r.read())["can"]["lamps_learnt"] == 2
    finally:
        srv.shutdown()
        srv.server_close()

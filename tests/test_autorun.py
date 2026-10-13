"""Picking frames without being told which: the automatic run's own judgement.

:func:`watch` sees a stream of frames after a state is sent and has to say
when the cluster has reacted and settled, and which frames to keep: the
latency before anything changes, a needle that sweeps on for a while, a lamp
that flashes and so never stands still, and a state that changes nothing.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from layoutval import autorun as A

W, H = 480, 200


def screen(lamp: bool = False, needle: float = 0.0, flash_on: bool = True) -> np.ndarray:
    img = np.full((H, W), 12, np.uint8)
    cv2.putText(img, "88", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 200, 3)
    angle = np.deg2rad(200 - 220 * needle)
    cx, cy = 300, 150
    cv2.line(img, (cx, cy), (int(cx + 80 * np.cos(angle)), int(cy - 80 * np.sin(angle))), 230, 3)
    if lamp and flash_on:
        cv2.circle(img, (420, 40), 12, 220, -1)
    return img


def stream(states, fps: float = 8.0, noise: float = 1.5, seed: int = 0):
    """(time, small, keep) for a function of time ``states(t)``."""
    rng = np.random.default_rng(seed)
    t = 0.0
    while True:
        img = states(t).astype(np.float32) + rng.normal(0, noise, (H, W))
        img = np.clip(img, 0, 255).astype(np.uint8)
        yield t, cv2.GaussianBlur(img, (3, 3), 0), (t, img)
        t += 1.0 / fps


def blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_a_lamp_that_lights_after_a_delay_is_waited_for_and_shot_steady():
    rest = screen()
    got = A.watch(stream(lambda t: screen(lamp=t >= 0.4)), [blur(rest)], 10.0)
    assert got.changed and got.settled and not got.flashing
    assert len(got.shots) == A.SHOTS_STEADY
    assert all(t >= 0.4 for t, _ in got.shots)


def test_a_needle_is_waited_out_until_it_stops():
    rest = screen()
    # Sweeps from 0 to 0.7 over a second and a half, then holds.
    got = A.watch(stream(lambda t: screen(needle=min(0.7, 0.7 * t / 1.5))), [blur(rest)], 10.0)
    assert got.changed and got.settled
    assert all(t >= 1.5 for t, _ in got.shots), [t for t, _ in got.shots]


def test_a_flashing_lamp_is_shot_across_its_flash():
    rest = screen()
    period = 0.76
    got = A.watch(stream(lambda t: screen(lamp=t >= 0.2, flash_on=(t % period) < period / 2)),
                  [blur(rest)], 10.0)
    assert got.changed and got.flashing and got.settled
    lit = [img[40, 420] > 120 for _, img in got.shots]
    assert any(lit) and not all(lit), lit
    assert len(got.shots) == A.SHOTS_FLASHING


def test_a_state_that_changes_nothing_costs_the_reaction_time_and_no_more():
    rest = screen()
    got = A.watch(stream(lambda t: rest), [blur(rest)], 10.0)
    assert not got.changed
    assert got.shots[0][0] == pytest.approx(A.REACT_S, abs=0.2)
    assert got.seconds < A.REACT_S + A.STEADY_S + 0.3


def test_back_from_a_flash_is_a_change_only_from_both_of_its_halves():
    lit_frames = [blur(screen(lamp=True)), blur(screen(lamp=True, flash_on=False))]
    # Back at rest looks like the flash's dark half: no change seen, no frame
    # of the flash taken for the rest screen.
    got = A.watch(stream(lambda t: screen()), lit_frames, 10.0)
    assert not got.changed
    assert all(img[40, 420] < 60 for _, img in got.shots)


def test_noise_level_stays_above_a_still_screens_flicker():
    rng = np.random.default_rng(1)
    base = screen().astype(np.float32)
    frames = [blur(np.clip(base + rng.normal(0, 3.0, base.shape), 0, 255).astype(np.uint8))
              for _ in range(4)]
    level = A.noise_level(frames)
    assert all(A.changed_px(a, b, level) < A.CHANGE_PX for a in frames for b in frames)
    assert A.changed_px(frames[0], blur(screen(lamp=True)), level) >= A.CHANGE_PX


def test_a_backlights_banding_is_not_a_change_and_a_lamp_through_it_is():
    # PWM through a camera: each row's gain moves a few percent from frame to
    # frame -- on white text, tens of levels.
    base = screen().astype(np.float32)
    rows = np.arange(H, dtype=np.float32)[:, None]

    def banded(img, phase):
        return blur(np.clip(img * (1 + 0.08 * np.sin(2 * np.pi * (rows / 40 + phase))), 0, 255)
                    .astype(np.uint8))

    frames = [banded(base, ph) for ph in (0.0, 0.37, 0.74, 0.11)]
    level = A.noise_level(frames)
    assert level < A.NOISE_MAX
    assert all(A.changed_px(a, b, level) < A.CHANGE_PX for a in frames for b in frames)
    lit = banded(screen(lamp=True).astype(np.float32), 0.5)
    assert A.changed_px(frames[0], lit, level) >= A.CHANGE_PX


def test_where_the_screen_lit_up_and_whether_an_outline_holds_it():
    dark = np.zeros((300, 400, 3), np.uint8)
    on = dark.copy()
    cv2.rectangle(on, (100, 80), (300, 220), (40, 40, 40), -1)      # the screen, lit
    cv2.circle(on, (200, 150), 30, (230, 230, 230), -1)
    lit = A.lit_region(dark, on)
    assert lit[150, 200] and not lit[20, 20]
    good = np.array([[95, 75], [305, 75], [305, 225], [95, 225]], float)
    small = np.array([[150, 120], [250, 120], [250, 180], [150, 180]], float)
    assert A.contains(good, lit) > 0.99
    assert A.contains(small, lit) < 0.6
    box = A.content_corners(lit)
    assert A.contains(box, lit) > 0.99


def test_the_display_size_from_its_shape():
    # Square on, 8:3: a cluster's 1920 x 720.
    corners = np.array([[100, 300], [1700, 300], [1700, 900], [100, 900]], float)
    size, why = A.estimate_size(corners, (1920, 1200))
    assert size == (1920, 720), why
    # A shape no common resolution has: its own, as many pixels across as seen.
    corners = np.array([[100, 300], [1500, 300], [1500, 980], [100, 980]], float)
    (w, h), why = A.estimate_size(corners, (1920, 1200))
    assert w == 1400 and abs(w / h - 1400 / 680) < 0.01, (w, h, why)


def test_the_bulb_check_is_the_start_up_frames_with_every_lamp_lit():
    run = np.zeros((200, 600, 3), np.uint8)
    cv2.putText(run, "0 km/h", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)

    def lamps(n):
        img = run.copy()
        for k in range(n):
            cv2.circle(img, (40 + 40 * k, 40), 12, (40, 60, 230), -1)
        return img

    logo = run.copy()
    cv2.putText(logo, "HI", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    frames = [np.zeros_like(run), logo, logo, lamps(12), lamps(12), lamps(12), run, run]
    idx, counts = A.bulb_frames(run, frames, 20.0)
    assert idx == [3, 4, 5], counts


# -- the whole run, against a small cluster of its own -------------------------
#
# A cluster that behaves as production ones do, in miniature: dark with the
# ignition off; with it on, a logo, then the bulb check (every lamp lit), then
# the running screen. Each lamp has a slot of its own. ABS is Off/On/Flash;
# the belt lamp lights while unbuckled; low fuel lights below 12 %, from the
# fuel level, which no lamp request carries; the brake lamp is lit only in
# the bulb check -- nothing on this bus drives it, so it is an unknown lamp.

MINI_DBC = """VERSION ""
NS_ :
BS_:
BU_: BCM ESC SRS ECM KOMBI

BO_ 100 ESC_1: 4 ESC
 SG_ ESC_Speed : 0|12@1+ (0.1,0) [0|409.5] "km/h" KOMBI
 SG_ ESC_ABS_TtReq : 12|2@1+ (1,0) [0|3] "" KOMBI
BO_ 200 SRS_1: 1 SRS
 SG_ SRS_BuckleDriver : 0|2@1+ (1,0) [0|3] "" KOMBI
BO_ 300 ECM_1: 2 ECM
 SG_ ECM_FuelLevel : 0|8@1+ (0.5,0) [0|127.5] "%" KOMBI
 SG_ ECM_Nothing : 8|1@1+ (1,0) [0|1] "" KOMBI
BO_ 400 BCM_1: 1 BCM
 SG_ BCM_KL15 : 0|2@1+ (1,0) [0|3] "" KOMBI

CM_ SG_ 100 ESC_ABS_TtReq "ABS warning lamp";
BA_DEF_ BO_ "GenMsgCycleTime" INT 0 65535;
BA_DEF_ SG_ "GenSigStartValue" INT 0 65535;
BA_DEF_DEF_ "GenMsgCycleTime" 100;
BA_DEF_DEF_ "GenSigStartValue" 0;
BA_ "GenSigStartValue" SG_ 200 SRS_BuckleDriver 1;
BA_ "GenSigStartValue" SG_ 300 ECM_FuelLevel 100;
BA_ "GenSigStartValue" SG_ 400 BCM_KL15 2;
VAL_ 100 ESC_ABS_TtReq 0 "Off" 1 "On" 2 "Flash" 3 "SNA" ;
VAL_ 200 SRS_BuckleDriver 0 "Unbuckled" 1 "Buckled" 2 "Fault" 3 "SNA" ;
VAL_ 400 BCM_KL15 0 "Off" 1 "Acc" 2 "On" 3 "Start" ;
"""

MDS = (640, 240)
SLOT, PITCH, ROW = 30, 44, 34
AMBER, RED = (20, 160, 245), (40, 50, 235)


def _lamp(img, k, draw, colour):
    x0 = MDS[0] // 2 - (4 * PITCH - (PITCH - SLOT)) // 2 + k * PITCH
    tile = np.zeros((SLOT, SLOT, 3), np.uint8)
    draw(tile, colour)
    sl = img[ROW - SLOT // 2:ROW + SLOT // 2, x0:x0 + SLOT]
    np.maximum(sl, tile, out=sl)


def _abs(g, c):
    cv2.circle(g, (15, 15), 12, c, 2, cv2.LINE_AA)
    cv2.putText(g, "A", (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


def _belt(g, c):
    cv2.circle(g, (15, 6), 4, c, -1)
    cv2.line(g, (15, 10), (15, 22), c, 4)
    cv2.line(g, (8, 12), (22, 24), c, 2)


def _fuel(g, c):
    cv2.rectangle(g, (7, 8), (19, 26), c, 2)
    cv2.line(g, (19, 12), (25, 18), c, 2)


def _brake(g, c):
    cv2.circle(g, (15, 15), 11, c, 2, cv2.LINE_AA)
    cv2.line(g, (15, 9), (15, 17), c, 3)
    cv2.circle(g, (15, 21), 2, c, -1)


class MiniCluster:
    """The tester's side (apply, ignition, quiet, rest) and the screen it makes."""

    bus_spec = {"interface": "fake"}
    FLASH = 0.3

    def __init__(self, rig):
        self.rig = rig
        self.values = dict(rig.rest)
        self.on_since = time.monotonic()
        self.on = True
        self.awake = True

    def apply(self, values):
        self.values = {**self.rig.rest, **values}

    def rest(self):
        self.apply({})

    def ignition(self, on):
        self.values[self.rig.ignition["signal"]] = self.rig.ignition["on" if on else "off"]
        self._power()

    def quiet(self, silent):
        self.awake = not silent
        self._power()

    def _power(self):
        on = self.awake and self.values.get("BCM_KL15") in ("On", "Start")
        if on and not self.on:
            self.on_since = time.monotonic()
        self.on = on

    def screen(self, faults=()):
        img = np.zeros((MDS[1], MDS[0], 3), np.uint8)
        if not self.on:
            return img
        img[:] = (14, 12, 10)
        cv2.rectangle(img, (2, 2), (MDS[0] - 3, MDS[1] - 3), (60, 60, 60), 2)
        age = time.monotonic() - self.on_since
        if age < 0.4:
            cv2.putText(img, "HELLO", (240, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            return img
        v = self.values
        speed = float(v.get("ESC_Speed", 0))
        cv2.putText(img, f"{min(speed, 260):.0f}", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (235, 235, 235), 3, cv2.LINE_AA)
        cv2.putText(img, "km/h", (70, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.circle(img, (470, 160), 55, (90, 80, 70), 2, cv2.LINE_AA)
        check = age < 1.4
        flash_on = (time.monotonic() % (2 * self.FLASH)) < self.FLASH
        abs_req = v.get("ESC_ABS_TtReq", "Off")
        if check or abs_req == "On" or (abs_req == "Flash" and flash_on):
            if "abs" not in faults:
                _lamp(img, 0, _abs, AMBER)
        if check or v.get("SRS_BuckleDriver") == "Unbuckled":
            _lamp(img, 1, _belt, RED)
        if check or float(v.get("ECM_FuelLevel", 100)) < 12:
            _lamp(img, 2, _fuel, AMBER)
        if check:
            _lamp(img, 3, _brake, RED)
        return img


def test_a_page_that_does_not_set_the_camera_leaves_it_as_it_is(tmp_path, monkeypatch):
    # An older page, or a browser without the controls: frames keep coming,
    # never with the settings asked for. The run goes on without them.
    from types import SimpleNamespace

    monkeypatch.setattr(A, "CAMERA_ACK_S", 0.5)
    session = SimpleNamespace(lamp_lib=None, reference=None, out_dir=tmp_path,
                              camera_settings={})
    caps = {"exposureMode": ["continuous", "manual"],
            "exposureTime": {"min": 1, "max": 2000, "step": 1}}
    run = A.AutoRun(session, camera={"caps": caps, "settings": {"exposureTime": 260}})
    run._display = lambda f: f
    jpg = cv2.imencode(".jpg", cv2.cvtColor(screen(), cv2.COLOR_GRAY2BGR))[1].tobytes()
    stop = threading.Event()

    def page():
        while not stop.is_set():
            run.feed.put(jpg, 0)                 # always the settings it started with
            time.sleep(0.03)

    threading.Thread(target=page, daemon=True).start()
    try:
        assert run._tune() == ""
        with pytest.raises(A.CameraIgnored):
            run._camera_look({"exposureTime": 80})
    finally:
        stop.set()
    assert run.summary["camera"]["tuned"] is False and "did not" in run.summary["camera"]["why"]
    assert session.camera_settings == {}


@pytest.fixture
def quick(monkeypatch):
    for name, value in {"REACT_S": 0.6, "SETTLE_S": 0.12, "SETTLE_SPAN_S": 0.8,
                        "STEADY_S": 0.35, "WINDOW_S": 1.0, "DARK_STILL_S": 0.4,
                        "RUN_STILL_S": 1.6, "MAX_SETTLE_S": 2.0}.items():
        monkeypatch.setattr(A, name, value)
    # watch()'s defaults were bound when it was defined.
    real = A.watch

    def watch(*a, **k):
        for key, name in (("react_s", "REACT_S"), ("settle_s", "SETTLE_S"),
                          ("span_s", "SETTLE_SPAN_S"), ("steady_s", "STEADY_S"),
                          ("window_s", "WINDOW_S"), ("max_settle_s", "MAX_SETTLE_S")):
            k.setdefault(key, getattr(A, name))
        return real(*a, **k)
    monkeypatch.setattr(A, "watch", watch)


def test_start_camera_and_everything_else_is_done(tmp_path, quick):
    pytest.importorskip("cantools")
    from layoutval.canbus import Rig
    from layoutval.discover import Discovery
    from layoutval.server import CaptureSession
    from layoutval.simulator import VirtualCamera

    dbc = tmp_path / "mini.dbc"
    dbc.write_text(MINI_DBC)
    rig = Rig.detect(dbc)
    assert rig.ignition["signal"] == "BCM_KL15" and rig.cluster_node == "KOMBI"
    cluster = MiniCluster(rig)
    cam = VirtualCamera(display_size=MDS, sensor_size=(900, 500), sampling_ratio=1.25,
                        tilt_deg=2.0, roll_deg=0.6, k1=0.0, k2=0.0, defocus_sigma=0.7,
                        noise_sigma=1.0, pwm_amplitude=0.05, seed=3)
    s = CaptureSession(tmp_path / "out", calib_mode="auto", size_from_phone=True,
                       can=cluster, discovery=Discovery(rig))
    faults: list[str] = []
    stop = threading.Event()
    # A webcam left on automatic: exposed for the room, so the screen is blown.
    caps = {"exposureMode": ["continuous", "manual"],
            "exposureTime": {"min": 1, "max": 2000, "step": 1}}
    webcam = {"exposureMode": "continuous", "exposureTime": 260}
    applied = {"seq": 0}

    def expose(img):
        f = img.astype(np.float32) * (webcam["exposureTime"] / 100.0)
        f += 0.8 * cv2.GaussianBlur(np.clip(f - 200, 0, None), (0, 0), 4)    # bloom
        return np.clip(f, 0, 255).astype(np.uint8)

    def pump():                        # the page: frames, as fast as it can
        while not stop.is_set():
            ok, jpg = cv2.imencode(".jpg", expose(cam.shoot(cluster.screen(faults))),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            try:
                p = s.auto_frame(jpg.tobytes(), applied["seq"], dict(webcam))
            except ValueError:
                time.sleep(0.02)
                continue
            if p.get("camera") and p["camera"]["seq"] > applied["seq"]:
                webcam.update(p["camera"]["set"])
                applied["seq"] = p["camera"]["seq"]
            time.sleep(0.02)

    def run(**kw):
        applied["seq"] = 0
        s.auto_start(camera={"caps": caps, "settings": dict(webcam)}, **kw)
        deadline = time.monotonic() + 240
        while not s.auto.progress.done and time.monotonic() < deadline:
            time.sleep(0.2)
        assert s.auto.progress.done, s.auto.progress.to_dict()
        return s.auto.progress.record, s.auto.summary

    threading.Thread(target=pump, daemon=True).start()
    try:
        rec, summary = run()
        # Taught: a later run tests against it, and finds a lamp that has gone.
        assert s.auto.teach is True
        rec2, summary2 = run()
        faults.append("abs")
        rec3, summary3 = run()
    finally:
        stop.set()
    rows = {r["id"]: r for r in rec["rows"]}
    assert rows["Display"]["verdict"] == "PASS", rows["Display"]
    # The camera was set up for the display, and is held there.
    assert summary["camera"]["tuned"] and rows["Camera"]["verdict"] == "PASS", summary["camera"]
    assert s.camera_settings["exposureMode"] == "manual"
    assert s.camera_settings["exposureTime"] < 200, s.camera_settings
    lamps = summary["teach"]["lamps"]
    assert "ABS warning lamp" in lamps, lamps
    assert any(n.startswith("Buckle driver") for n in lamps), lamps
    assert "Fuel level low" in lamps, lamps
    # Seen in the bulb check, lit by nothing on the bus.
    assert len(summary["teach"]["unknown_lamps"]) == 1, summary["teach"]
    assert summary["teach"]["layout"] == "fixed slots"
    assert "ECM_Nothing" in " ".join(summary["teach"]["no_effect"]) or \
        "Nothing" in " ".join(summary["teach"]["no_effect"]), summary["teach"]["no_effect"]
    assert rows["Lamp check"]["verdict"] == "PASS", [r for r in rec["rows"] if r["verdict"] != "PASS"]
    assert rows["Layout"]["verdict"] == "PASS", rows["Layout"]
    assert s.lamp_lib is not None and (s.auto.folder / "auto.json").is_file()

    # The second run tests: no teaching, the bulb check and every lamp as taught,
    # with the camera given the settings it was taught with.
    assert summary2["mode"] == "test" and "teach" not in summary2
    assert summary2["camera"] == {"settings": s.camera_settings, "tuned": False}
    rows2 = {r["id"]: r for r in rec2["rows"]}
    assert rows2["Bulb check"]["verdict"] == "PASS", rec2["rows"]
    assert rows2["Lamp check"]["verdict"] == "PASS", rec2["rows"]
    assert rows2["Layout"]["verdict"] == "PASS", rows2["Layout"]
    # ABS no longer drawn: missing in the bulb check and in each of its states.
    fails = {(r["id"], r["detail"].split(" (")[0]) for r in rec3["rows"] if r["verdict"] == "FAIL"}
    assert ("ABS warning lamp", "missing") in fails, rec3["rows"]
    assert rec3["verdict"] == "FAIL"
    assert {r["id"] for r in rec3["rows"] if r["verdict"] == "FAIL"} <= {
        "ABS warning lamp", "Bulb check", "Lamp check"}, rec3["rows"]

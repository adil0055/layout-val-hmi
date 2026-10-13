"""Setting a webcam up for a display, against a webcam that behaves like one.

The fake camera starts as a real one left on automatic does in front of a
lit display: exposed for the room, so the screen is blown and blooms; a
colour cast; focus off; sharpening on.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from layoutval import camtune as C

TRUE_K = 5200.0          # the display's white, as the camera would need to be told
BEST_FOCUS = 0.42

CAPS = {
    "exposureMode": ["continuous", "manual"], "exposureTime": {"min": 1, "max": 5000, "step": 1},
    "whiteBalanceMode": ["continuous", "manual"], "colorTemperature": {"min": 2800, "max": 6500, "step": 10},
    "focusMode": ["continuous", "manual"], "focusDistance": {"min": 0.0, "max": 1.0, "step": 0.01},
    "sharpness": {"min": 0, "max": 7, "step": 1},
}
AUTO = {"exposureMode": "continuous", "exposureTime": 260, "whiteBalanceMode": "continuous",
        "colorTemperature": 6500, "focusMode": "continuous", "focusDistance": 0.9, "sharpness": 5}


def display() -> np.ndarray:
    img = np.full((360, 640, 3), 12, np.uint8)
    cv2.circle(img, (200, 190), 140, (90, 80, 70), 2, cv2.LINE_AA)
    for a in range(0, 271, 15):
        t = np.deg2rad(225 - a)
        p1 = (int(200 + 120 * np.cos(t)), int(190 - 120 * np.sin(t)))
        p2 = (int(200 + 135 * np.cos(t)), int(190 - 135 * np.sin(t)))
        cv2.line(img, p1, p2, (230, 230, 230), 3, cv2.LINE_AA)
    cv2.putText(img, "120", (150, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (235, 235, 235), 4, cv2.LINE_AA)
    cv2.putText(img, "Trip 0.0 km", (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.circle(img, (480, 260), 18, (40, 50, 235), -1)                      # a red lamp
    return img


class FakeWebcam:
    def __init__(self):
        self.scene = display().astype(np.float32)
        self.settings = dict(AUTO)
        self.looks = 0

    def shoot(self, s: dict) -> np.ndarray:
        gain = s["exposureTime"] / 100.0 * s.get("iso", 100) / 100.0
        img = self.scene * gain
        over = np.clip(img - 200, 0, None)
        img = img + 0.8 * cv2.GaussianBlur(over, (0, 0), 6)                  # bloom
        k = s["colorTemperature"] / TRUE_K
        img = img * np.array([k ** -0.8, 1.0, k ** 0.8], np.float32)        # BGR: warm or cool cast
        sigma = 0.3 + 8.0 * abs(s["focusDistance"] - BEST_FOCUS)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        amount = 1.5 * s["sharpness"] / 7.0
        img = img + amount * (img - cv2.GaussianBlur(img, (0, 0), 2.0))     # sharpening halos
        img += np.random.default_rng(self.looks).normal(0, 1.2, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def look(self, settings: dict) -> list[np.ndarray]:
        self.settings.update(settings)
        self.looks += 1
        return [self.shoot(self.settings) for _ in range(2)]


def test_a_webcam_left_on_auto_is_set_up_for_the_display():
    cam = FakeWebcam()
    before = cam.shoot(cam.settings)
    level0, clip0 = C.brightest(before)
    assert clip0 > 0.01                                     # blown, as it starts
    tuner = C.Tuner(CAPS, AUTO, cam.look)
    final = tuner.run()
    after = cam.shoot(cam.settings)
    level, clip = C.brightest(after)
    assert C.BAND[0] <= level <= C.BAND[1] and clip <= C.CLIP_MAX, (level, clip, tuner.steps)
    assert abs(C.white_balance(after) - 1.0) < 0.08, tuner.steps
    assert abs(final["focusDistance"] - BEST_FOCUS) < 0.1, tuner.steps
    assert final["sharpness"] == 0
    assert final["exposureMode"] == final["whiteBalanceMode"] == final["focusMode"] == "manual"
    # Held: given back on a later run, the camera sees the display the same way.
    assert C.brightest(cam.shoot({**AUTO, **final}))[0] == pytest.approx(level, abs=3)
    assert cam.looks < 40


def test_a_camera_that_only_takes_an_exposure_compensation():
    cam = FakeWebcam()
    caps = {"exposureCompensation": {"min": -3, "max": 3, "step": 0.1}}

    def look(s):
        # Compensation in stops on top of the camera's own (blown) choice.
        return cam.look({"exposureTime": 260 * 2 ** s.get("exposureCompensation", 0.0)})

    final = C.Tuner(caps, {"exposureCompensation": 0.0}, look).run()
    level, clip = C.brightest(cam.shoot(cam.settings))
    assert final["exposureCompensation"] < 0 and clip <= C.CLIP_MAX, (final, level, clip)


def test_a_camera_with_nothing_to_set_is_left_alone():
    calls = []
    assert C.Tuner({}, {}, lambda s: calls.append(s) or [display()]).run() == {}
    assert not calls
    assert C.controls(CAPS) == ["exposure", "white balance", "focus", "sharpness"]
    assert C.controls({**CAPS, "iso": {"min": 50, "max": 800}})[-1] == "gain"


def test_a_lamp_in_view_that_is_not_the_display_does_not_set_the_exposure():
    # A room's lamp, or its reflection on the glass, inside the display's
    # outline: it clips at any exposure a display wants. Measured with it,
    # the exposure is cut until it does not, and the display goes dark.
    class Lamp(FakeWebcam):
        def shoot(self, s):
            img = super().shoot(s).astype(np.float32)
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            blob = np.exp(-((xx - 600) ** 2 + (yy - 40) ** 2) / (2 * 9.0 ** 2))
            img += (3000.0 * s["exposureTime"] / 100.0 * blob)[..., None]
            return np.clip(img, 0, 255).astype(np.uint8)

    content = cv2.dilate((display().max(axis=2) > 30).astype(np.uint8),
                         np.ones((9, 9), np.uint8)) > 0
    assert not content[40, 600]
    caps = {k: CAPS[k] for k in ("exposureMode", "exposureTime")}
    cam = Lamp()
    C.Tuner(caps, AUTO, cam.look, mask=content).run()
    level, clip = C.brightest(cam.shoot(cam.settings), content)
    assert C.BAND[0] <= level <= C.BAND[1] and clip <= C.CLIP_MAX, (level, clip)
    # The same without being told what the display is: turned right down.
    blind = Lamp()
    C.Tuner(caps, AUTO, blind.look).run()
    assert blind.settings["exposureTime"] < 0.5 * cam.settings["exposureTime"]


def test_gain_goes_down_before_the_exposure_does():
    # Left on automatic in a dim room, the camera's gain is up. As a camera
    # does in a bright scene: gain to its lowest, so the exposure the display
    # needs is the longest it can be.
    cam = FakeWebcam()
    caps = {**CAPS, "iso": {"min": 100, "max": 1600, "step": 100}}
    start = {**AUTO, "iso": 800}
    final = C.Tuner(caps, start, cam.look).run()
    level, clip = C.brightest(cam.shoot(cam.settings))
    assert final["iso"] == 100 and C.BAND[0] <= level <= C.BAND[1] and clip <= C.CLIP_MAX
    held_gain = FakeWebcam()
    C.Tuner(CAPS, start, held_gain.look).run()          # no gain control: left at 800
    assert final["exposureTime"] > 6 * held_gain.settings["exposureTime"]

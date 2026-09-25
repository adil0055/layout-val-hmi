"""Camera shake: a photograph smeared during the exposure.

A hand-held phone moves a little while the shutter is open, and the photograph
is the scene convolved with the path it took -- a short streak, a few pixels
long. Between two photographs, that makes one sharper than the other, and
correlation, which the measurement rests on, reads a sharp telltale against a
smeared one as different content. Measured on two bench photographs: a 3 px
smear put 3-19 elements of a good screen at FAIL, "wrong content", and a 6 px
smear most of them.

Moving the camera *between* shots is a different thing and is already handled
by re-solving the pose. This is blur *within* a shot, and the answer to it is
to compare like with like: work out the smear from the two photographs
themselves, and apply it to the sharper one.

**Working it out.** The two frames are aligned in display space, so the blurred
one is, to a good approximation, the sharp one convolved with an unknown kernel.
With the sharp one known, that kernel is a regularised division in the Fourier
domain -- the textbook Wiener estimate -- cut to a small support, cleaned of
negative lobes and normalised. Both directions are tried, because either photo
may be the shaken one, and the kernel is used only if applying it actually
brings the two frames closer.

**Why it cannot hide a fault.** The kernel is one for the whole frame, fitted to
every element at once, so an element that moved does not move it. And it is
re-centred on its own centroid before use, so blurring introduces no shift at
all: it can change how sharp an element looks, never where it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from layoutval.capture import to_gray

RADIUS = 20
"""Largest smear handled, in display pixels either side of centre."""


@dataclass
class ShakeMatch:
    reference: np.ndarray
    live: np.ndarray
    kernel: np.ndarray | None
    """The smear applied, centred; None when neither photo was blurred more."""
    applied_to: str
    """"reference", "live" or "" -- which photo was blurred to match the other."""
    spread_px: float
    """RMS radius of the kernel: how far the smear reaches."""


def _estimate(sharp: np.ndarray, blurred: np.ndarray, radius: int) -> np.ndarray | None:
    """Kernel k with blurred ~= k * sharp, cut to ``radius``; None if there is none."""
    a = sharp - sharp.mean()
    b = blurred - blurred.mean()
    h, w = a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    A = cv2.dft(a * win, flags=cv2.DFT_COMPLEX_OUTPUT)
    B = cv2.dft(b * win, flags=cv2.DFT_COMPLEX_OUTPUT)
    ar, ai = A[..., 0], A[..., 1]
    br, bi = B[..., 0], B[..., 1]
    power = ar * ar + ai * ai
    eps = 1e-2 * float(power.mean())
    # B * conj(A) / (|A|^2 + eps)
    num = np.dstack([br * ar + bi * ai, bi * ar - br * ai])
    H = num / (power + eps)[..., None]
    k = cv2.idft(H.astype(np.float32), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    k = np.fft.fftshift(k)
    cy, cx = h // 2, w // 2
    k = k[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].astype(np.float64)
    k[k < 0.05 * k.max()] = 0.0
    total = k.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    k /= total
    # Keep only the part connected to the peak: the smear is one streak, and
    # isolated specks further out are the estimate's noise.
    mask = (k > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if n > 2:
        py, px = np.unravel_index(int(np.argmax(k)), k.shape)
        k[labels != labels[py, px]] = 0.0
        k /= k.sum()
    return _centred(k)


def _centred(k: np.ndarray) -> np.ndarray:
    """``k`` moved so its centroid is exactly at its centre: a blur, not a shift."""
    ys, xs = np.mgrid[0:k.shape[0], 0:k.shape[1]]
    c = (k.shape[1] - 1) / 2.0, (k.shape[0] - 1) / 2.0
    dx, dy = c[0] - float((k * xs).sum()), c[1] - float((k * ys).sum())
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(k.astype(np.float32), m, (k.shape[1], k.shape[0]),
                         flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float64)
    return out / out.sum()


def _spread(k: np.ndarray) -> float:
    ys, xs = np.mgrid[0:k.shape[0], 0:k.shape[1]]
    cx, cy = float((k * xs).sum()), float((k * ys).sum())
    return float(np.sqrt((k * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum()))


def match_shake(reference: np.ndarray, live: np.ndarray, *, radius: int = RADIUS,
                min_gain: float = 0.1) -> ShakeMatch:
    """Blur whichever frame is sharper by the smear that separates them.

    Both frames must already be aligned in display space. Returns them
    unchanged when neither is measurably more blurred than the other, or when
    applying the estimated smear would not bring them at least ``min_gain``
    (10%) closer.
    """
    a = to_gray(reference).astype(np.float32)
    b = to_gray(live).astype(np.float32)
    if a.shape != b.shape or min(a.shape) < 4 * radius:
        return ShakeMatch(reference, live, None, "", 0.0)

    def gap(x: np.ndarray, y: np.ndarray) -> float:
        d = (x - x.mean()) - (y - y.mean())
        return float(np.sqrt((d * d).mean()))

    best = ("", None, gap(a, b))
    for name, sharp, blurred in (("reference", a, b), ("live", b, a)):
        k = _estimate(sharp, blurred, radius)
        if k is None or _spread(k) < 0.5:
            continue
        g = gap(cv2.filter2D(sharp, -1, k.astype(np.float32)), blurred)
        if g < best[2]:
            best = (name, k, g)
    name, k, g = best
    if k is None or g > (1.0 - min_gain) * gap(a, b):
        return ShakeMatch(reference, live, None, "", 0.0)
    kf = k.astype(np.float32)
    if name == "reference":
        return ShakeMatch(cv2.filter2D(reference, -1, kf), live, k, name, _spread(k))
    return ShakeMatch(reference, cv2.filter2D(live, -1, kf), k, name, _spread(k))

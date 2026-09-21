"""Stages 2 and 3 -- undistortion and rectification into display space.

This is where the accuracy budget lives.  Get the geometry right and a
twenty-line function measures position to a tenth of a pixel; get it wrong and
no model recovers it.

Three ways to obtain the homography ``H`` mapping display coordinates to camera
coordinates, in order of preference:

A. :func:`homography_from_display_pattern` -- the cluster renders a calibration
   pattern itself.  Direct, exact correspondence between framebuffer and camera
   coordinates.  Ask the HMI team for a layout-calibration screen in test
   builds; it is the highest-leverage thing you can get and it pays for itself
   permanently.
B. :func:`homography_from_charuco` -- a ChArUco board on the bezel, plus a
   one-time measurement of where the active area sits relative to it.
C. :func:`homography_from_display_edges` -- fit lines to the display's own
   edges on a full-white screen.  Workable fallback, least stable, because it
   re-derives the geometry from content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from layoutval.capture import to_gray


# --------------------------------------------------------------------------
# intrinsics
# --------------------------------------------------------------------------


@dataclass
class Intrinsics:
    """Camera matrix and distortion coefficients.

    Without undistortion, elements near the frame edges read as shifted on a
    perfectly good cluster -- the error is largest exactly where lens distortion
    is largest, which makes it look like a real layout defect in one corner.
    """

    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    rms: float = 0.0
    """Reprojection error from the calibration solve, in camera pixels.  Below
    0.3 px is the gate for accepting a calibration."""

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    def undistort_maps(self, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precomputed remap tables plus the new camera matrix.

        ``alpha=0`` crops to the all-valid region, which is what you want: it
        keeps every pixel in the undistorted frame meaningful.
        """
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, self.image_size, alpha, self.image_size
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, new_K, self.image_size, cv2.CV_32FC1
        )
        return map1, map2, new_K

    def to_dict(self) -> dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "image_size": list(self.image_size),
            "rms": self.rms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Intrinsics":
        return cls(
            K=np.array(d["K"], dtype=np.float64),
            dist=np.array(d["dist"], dtype=np.float64),
            image_size=tuple(d["image_size"]),
            rms=float(d.get("rms", 0.0)),
        )


class Undistorter:
    """Applies stored intrinsics.  Builds the remap tables once."""

    def __init__(self, intrinsics: Intrinsics, alpha: float = 0.0) -> None:
        self.intrinsics = intrinsics
        self.map1, self.map2, self.new_K = intrinsics.undistort_maps(alpha)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_CUBIC)


def board_points_for_intrinsics(
    image: np.ndarray,
    *,
    board: "cv2.aruco.CharucoBoard | None" = None,
    pattern_size: tuple[int, int] | None = None,
    square_size_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One view's (object points, image points), for accumulating a calibration.

    Pulled out of :func:`calibrate_intrinsics` so that views can be collected
    one at a time -- from a phone, over the network -- without holding every
    full-resolution frame in memory.  A phone photograph is tens of megabytes
    and a view's correspondences are a few kilobytes.

    Raises :class:`RuntimeError` when the board is not readable in this view,
    which is the common case and worth telling the person about while they are
    still standing in front of the cluster.
    """
    if board is not None:
        corners, ids = detect_charuco(image, board)
        obj = np.asarray(board.getChessboardCorners(), np.float32)[ids]
        return obj, corners.astype(np.float32)

    if pattern_size is None:
        raise ValueError("need either a ChArUco board or a chessboard pattern size")
    ok, corners = cv2.findChessboardCornersSB(
        to_gray(image), pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not ok:
        raise RuntimeError(
            f"no {pattern_size[0]}x{pattern_size[1]} chessboard in this view"
        )
    obj = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    obj *= square_size_mm
    return obj, corners.reshape(-1, 2).astype(np.float32)


@dataclass
class LensCheck:
    """Whether a lens solve can be trusted, which is not what rms tells you.

    Reprojection error measures how well the model fits the views it was given.
    Give it twelve similar views and it fits them beautifully and is wrong
    everywhere else -- and reports a *lower* rms for doing so.  Measured against
    the simulated bezel:

        how the lens was shot        rms     element error
        clean, varied poses        0.052          0.107 px
        soft focus                 0.227          0.520 px
        poses too similar          0.051          1.256 px
        similar + soft             0.205          2.977 px
        noisy + soft + similar     0.269          5.614 px
        (no undistortion at all)       -          4.059 px

    The last row is the one that matters: a solve passing a 0.3 px rms gate was
    worse than not undistorting at all, and nothing in its rms said so.  So
    ``rms`` is reported but not gated on.  Two things are:

    ``holdout_px``
        Fit on most of the views, measure reprojection on the ones held back.
        This asks whether the model *generalises* rather than whether it fits,
        which is exactly the question rms cannot answer.
    ``tilt_spread_deg``
        How much the board's orientation actually varied.  Views that all look
        alike cannot separate the lens from the pose, and this is the cause
        behind most of the bad rows above.
    """

    rms: float
    holdout_px: float
    tilt_spread_deg: float
    depth_spread: float
    views: int

    #: Below this the views did not really differ, and the solve is
    #: unconstrained however tight its rms looks. Measured: good sets ran 10
    #: degrees of spread and every bad one 2.1-2.6.
    MIN_TILT_SPREAD_DEG = 6.0

    #: Held-out reprojection maps almost linearly onto the error the rig then
    #: makes, measured by adding corner-localisation noise (which is what moire
    #: off a screen, JPEG sharpening and a soft board all amount to):
    #:
    #:     holdout px   element error
    #:          0.05         0.093 px
    #:          0.26         0.113 px
    #:          0.52         0.261 px
    #:          0.78         0.460 px
    #:          1.04         0.631 px
    #:          2.34         1.296 px
    #:
    #: So roughly ``0.6 * holdout``, and that is a number to report rather than
    #: a line to refuse at. Even the worst row above beats not undistorting,
    #: which on the same lens costs 4.08 px -- so a loose solve is still worth
    #: having, as long as nobody mistakes it for a tight one.
    ERROR_PER_HOLDOUT_PX = 0.6

    #: Past this the solve is too loose to be worth the undistortion pass.
    MAX_HOLDOUT_PX = 3.0

    def complaint(self) -> str | None:
        """What is wrong with this solve, in words, or None if nothing is.

        Two different failures, and neither is visible in rms alone: a solve
        that is unconstrained because the views were all alike, and a solve
        whose views were individually poor.
        """
        if self.tilt_spread_deg < self.MIN_TILT_SPREAD_DEG:
            return (
                f"the views were all shot from about the same angle "
                f"({self.tilt_spread_deg:.0f} degrees of spread). A lens cannot "
                "be separated from the pose that way, and the solve will look "
                "tight and be wrong -- measured, a set like this reported 0.05 "
                "px reprojection error and left 1.3 px of real error behind. "
                "Re-shoot with the board steeply angled in several directions"
            )
        if self.holdout_px > 3.0 * max(self.rms, 0.05) and self.holdout_px > 0.3:
            return (
                f"the solve fits the views it was given ({self.rms:.2f} px) far "
                f"better than views it was not ({self.holdout_px:.2f} px), which "
                "means it is describing these particular photographs rather "
                "than the lens. More varied angles and distances"
            )
        if self.holdout_px > self.MAX_HOLDOUT_PX:
            return (
                f"the lens model is {self.holdout_px:.2f} px out on views it did "
                f"not see, past the {self.MAX_HOLDOUT_PX:.1f} px where "
                "undistorting stops being worth doing. The views themselves "
                "were poor -- soft, noisy, or shot off a screen, where the "
                "display's own pixel grid beats against the sensor's and moves "
                "every corner. Print the board if you can, fill the frame with "
                "it, and keep it sharp"
            )
        return None

    def expected_element_error_px(self) -> float:
        """Roughly what this lens solve will cost the measurements through it.

        Not a guarantee -- a floor to set tolerances against, which is the same
        arithmetic :meth:`Tolerance.defensible_floor` does for everything else.
        A number that is stated can be worked with; the same number hidden
        behind a pass mark cannot.
        """
        return self.ERROR_PER_HOLDOUT_PX * max(self.holdout_px, 0.0)

    def quality(self) -> str:
        """One word for how good this solve is, for a record to carry."""
        implied = self.expected_element_error_px()
        if implied <= 0.15:
            return "tight"
        if implied <= 0.40:
            return "workable"
        return "loose"


def _reprojection_px(obj, img, K, dist, rvec, tvec) -> float:
    projected, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))


def check_intrinsics(
    obj_points: Sequence[np.ndarray],
    img_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    folds: int = 4,
) -> LensCheck:
    """Fit on most of the views and score the ones held back.

    The held-out number is the honest one: it says how far the lens model is
    from views it has never seen, which is what undistorting a new photograph
    actually asks of it.
    """
    n = len(obj_points)
    objs = [o.reshape(-1, 1, 3) for o in obj_points]
    imgs = [i.reshape(-1, 1, 2) for i in img_points]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objs, imgs, image_size, None, None)

    # How much the board's orientation varied. The rotation vector's magnitude
    # is the angle the board is turned through, so the spread of those is a
    # direct measure of how different the views really were.
    angles = [float(np.degrees(np.linalg.norm(r))) for r in rvecs]
    depths = [float(abs(t.ravel()[2])) for t in tvecs]
    tilt_spread = float(np.percentile(angles, 90) - np.percentile(angles, 10))
    depth_spread = (float(np.ptp(depths) / max(np.median(depths), 1e-9))
                    if depths else 0.0)

    held: list[float] = []
    for fold in range(folds):
        test = list(range(fold, n, folds))
        train = [i for i in range(n) if i not in test]
        if len(train) < 6 or not test:
            continue
        try:
            _, Kf, distf, _, _ = cv2.calibrateCamera(
                [objs[i] for i in train], [imgs[i] for i in train],
                image_size, None, None)
            for i in test:
                # Pose is re-solved per held-out view: we are scoring the lens,
                # not our ability to guess where the board was.
                ok, rvec, tvec = cv2.solvePnP(objs[i], imgs[i], Kf, distf)
                if ok:
                    held.append(_reprojection_px(
                        obj_points[i], img_points[i], Kf, distf, rvec, tvec))
        except cv2.error:
            continue

    return LensCheck(
        rms=float(rms),
        holdout_px=float(np.median(held)) if held else float("nan"),
        tilt_spread_deg=tilt_spread,
        depth_spread=depth_spread,
        views=n,
    )


def solve_intrinsics(
    obj_points: Sequence[np.ndarray],
    img_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    min_views: int = 8,
) -> Intrinsics:
    """Solve K and distortion from views already reduced to correspondences."""
    if len(obj_points) < min_views:
        raise RuntimeError(
            f"only {len(obj_points)} usable views; need at least {min_views} "
            "spanning different poses and depths"
        )
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [o.reshape(-1, 1, 3) for o in obj_points],
        [i.reshape(-1, 1, 2) for i in img_points],
        image_size, None, None,
    )
    return Intrinsics(K=K, dist=dist, image_size=image_size, rms=float(rms))


def calibrate_intrinsics(
    images: Sequence[np.ndarray],
    pattern_size: tuple[int, int],
    square_size_mm: float = 1.0,
    *,
    min_views: int = 8,
) -> Intrinsics:
    """Standard checkerboard intrinsic calibration.

    ``pattern_size`` is (inner corners per row, inner corners per column).
    Raises if too few views were usable -- a calibration solved from three
    near-identical views is worse than no calibration, because it looks fine.
    """
    if not images:
        raise ValueError("no calibration images")
    h, w = images[0].shape[:2]
    obj = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    obj *= square_size_mm

    obj_points, img_points = [], []
    for img in images:
        gray = to_gray(img)
        ok, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if not ok:
            continue
        obj_points.append(obj)
        img_points.append(corners.reshape(-1, 2).astype(np.float32))

    if len(obj_points) < min_views:
        raise RuntimeError(
            f"only {len(obj_points)} of {len(images)} calibration views were usable; "
            f"need at least {min_views} spanning different poses and depths"
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    return Intrinsics(K=K, dist=dist, image_size=(w, h), rms=float(rms))


# --------------------------------------------------------------------------
# display-space rectification
# --------------------------------------------------------------------------


@dataclass
class DisplayGeometry:
    """The mapping between display space and camera space.

    ``H`` maps *display* coordinates to *camera* coordinates.  Rectification is
    therefore the inverse warp, which :func:`cv2.warpPerspective` performs
    directly via ``WARP_INVERSE_MAP`` -- no explicit inversion, no extra
    numerical error.
    """

    H: np.ndarray
    display_size: tuple[int, int]
    method: str = "unknown"
    residual_px: float = 0.0
    """RMS reprojection residual of the homography fit, in camera pixels."""

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=np.float64).reshape(3, 3)
        self.display_size = (int(self.display_size[0]), int(self.display_size[1]))

    def rectify(self, img: np.ndarray, H: np.ndarray | None = None) -> np.ndarray:
        """Warp a camera frame into display space."""
        return cv2.warpPerspective(
            img,
            np.asarray(H if H is not None else self.H, dtype=np.float64),
            self.display_size,
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def display_to_camera(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def camera_to_display(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def sampling_ratio(self) -> float:
        """Camera pixels per display pixel, at the display centre.

        Below ~2 you cannot reliably resolve a one-display-pixel shift with
        sub-pixel interpolation; 3 is comfortable.  A 1080p camera on a
        1920-wide cluster samples at roughly 1:1, and the honest floor there is
        about +/-1 display pixel.  Know which regime you are in before promising
        anyone a tolerance.
        """
        w, h = self.display_size
        cx, cy = w / 2.0, h / 2.0
        p = self.display_to_camera(
            np.array([[cx, cy], [cx + 1.0, cy], [cx, cy + 1.0]], dtype=np.float64)
        )
        return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "H": self.H.tolist(),
            "display_size": list(self.display_size),
            "method": self.method,
            "residual_px": self.residual_px,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DisplayGeometry":
        return cls(
            H=np.array(d["H"], dtype=np.float64),
            display_size=tuple(d["display_size"]),
            method=d.get("method", "unknown"),
            residual_px=float(d.get("residual_px", 0.0)),
        )


@dataclass
class Calibration:
    """Everything the rig knows about itself, stored as one file.

    The rig settings live in here alongside the numbers because a calibration
    is only valid for the rig that produced it: change the lens, the standoff or
    the exposure and the stored geometry is quietly wrong.
    """

    intrinsics: Intrinsics | None
    geometry: DisplayGeometry
    rig: dict[str, Any] = field(default_factory=dict)
    drift_alarm_px: float = 2.0

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
                    "geometry": self.geometry.to_dict(),
                    "rig": self.rig,
                    "drift_alarm_px": self.drift_alarm_px,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(
            intrinsics=Intrinsics.from_dict(d["intrinsics"]) if d.get("intrinsics") else None,
            geometry=DisplayGeometry.from_dict(d["geometry"]),
            rig=d.get("rig", {}),
            drift_alarm_px=float(d.get("drift_alarm_px", 2.0)),
        )


# --------------------------------------------------------------------------
# homography solvers
# --------------------------------------------------------------------------


def _homography_residual(src: np.ndarray, dst: np.ndarray, H: np.ndarray) -> float:
    proj = cv2.perspectiveTransform(
        np.asarray(src, np.float64).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - np.asarray(dst, np.float64)) ** 2, axis=1))))


def chessboard_display_points(
    pattern_size: tuple[int, int],
    square_px: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Framebuffer coordinates of the inner corners of a rendered checkerboard.

    Inner corner ``(col, row)`` of a board whose first square starts at
    ``origin`` sits at ``origin + ((col+1)*square, (row+1)*square)``.
    """
    cols, rows = pattern_size
    pts = [
        (origin[0] + (c + 1) * square_px, origin[1] + (r + 1) * square_px)
        for r in range(rows)
        for c in range(cols)
    ]
    return np.array(pts, dtype=np.float64)


def _orient_corners(corners: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray:
    """Normalise the 180-degree ambiguity in chessboard corner ordering.

    ``findChessboardCornersSB`` returns corners in a consistent grid order but
    may start from either end.  Anchor on whichever end is closer to the image
    origin so the correspondence with the display points is stable across runs.
    """
    corners = corners.reshape(-1, 2)
    if np.sum(corners[0] ** 2) > np.sum(corners[-1] ** 2):
        corners = corners[::-1]
    return corners


def homography_from_display_pattern(
    camera_img: np.ndarray,
    pattern_size: tuple[int, int],
    display_points: np.ndarray,
    *,
    display_size: tuple[int, int],
    refine: bool = True,
) -> DisplayGeometry:
    """Method A -- the cluster renders the calibration pattern itself.

    ``display_points`` are the framebuffer coordinates of the pattern's inner
    corners, in the same row-major order the detector returns them
    (see :func:`chessboard_display_points`).
    """
    gray = to_gray(camera_img)
    ok, corners = cv2.findChessboardCornersSB(
        gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not ok:
        raise RuntimeError(
            "calibration pattern not found -- check focus, exposure and that the "
            "whole board is inside the frame"
        )
    corners = corners.astype(np.float32).reshape(-1, 1, 2)
    if refine:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4),
        )
    cam = _orient_corners(corners, pattern_size)
    disp = np.asarray(display_points, dtype=np.float64).reshape(-1, 2)
    if len(cam) != len(disp):
        raise ValueError(
            f"detector returned {len(cam)} corners but {len(disp)} display points were given"
        )
    H, _ = cv2.findHomography(disp, cam, method=0)
    if H is None:
        raise RuntimeError("homography solve failed")
    return DisplayGeometry(
        H=H,
        display_size=display_size,
        method="display_pattern",
        residual_px=_homography_residual(disp, cam, H),
    )


def homography_from_charuco(
    camera_img: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
    display_from_board: np.ndarray,
    *,
    display_size: tuple[int, int],
) -> DisplayGeometry:
    """Method B -- a ChArUco board fixed to the bezel.

    ChArUco rather than plain ArUco specifically because of corner accuracy: the
    ArUco squares give identity and occlusion tolerance, but the interpolated
    corners belong to a chessboard, and chessboard corners refine far more
    accurately.

    Marker corner refinement is **disabled** here on purpose.  The OpenCV
    documentation is explicit that when the result feeds a homography, the
    proximity of the chessboard squares makes the sub-pixel step deviate, and
    those deviations propagate into the interpolated corners.

    ``display_from_board`` is the 3x3 transform taking board-plane coordinates
    (the board's own units, as returned by ``board.getChessboardCorners``) into
    display coordinates.  Measure it once with method A and store it -- the
    bezel does not move relative to the active area.
    """
    cam, ids = detect_charuco(camera_img, board)
    board_pts = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
    src_board = board_pts[ids]

    disp = cv2.perspectiveTransform(
        src_board.reshape(-1, 1, 2), np.asarray(display_from_board, np.float64)
    ).reshape(-1, 2)

    H, _ = cv2.findHomography(disp, cam, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("homography solve failed")
    return DisplayGeometry(
        H=H,
        display_size=display_size,
        method="charuco_bezel",
        residual_px=_homography_residual(disp, cam, H),
    )


def homography_from_screen_content(
    render: np.ndarray,
    camera_frame: np.ndarray,
    *,
    display_size: tuple[int, int] | None = None,
    min_inliers: int = 12,
    ratio: float = 0.75,
    refine: bool = True,
    ecc_iterations: int = 400,
) -> DisplayGeometry:
    """Method D -- no pattern at all: match the screen against its own framebuffer.

    The other three methods need the cluster to draw something for the benefit
    of the camera.  This one does not: if the framebuffer is available -- and it
    is, on the machine running the HMI -- then the screen's own artwork is the
    calibration target.  ``render`` is that framebuffer, in display coordinates;
    ``camera_frame`` is a photograph of it.

    Two stages, and both are needed:

    * **SIFT and RANSAC** give the correspondence.  At matched scale that is
      already as accurate as a chessboard, but it degrades when the render and
      the photograph differ in scale or sharpness -- 0.05 px becomes 0.7 px on
      an under-sampled rig, because scale-invariant keypoints localise less
      precisely across a large scale ratio.
    * **ECC in homography mode**, seeded from that, closes it back up.  Measured
      against a chessboard across pose, noise, focus, sampling ratio and lens
      distortion, the pair stays within about 0.02 px of it everywhere; the
      seed on its own does not.

    SIFT is used rather than ORB because ORB's seed is far looser (1.16 px
    against 0.05) -- with ECC behind it either converges to the same answer, but
    a looser seed is likelier to converge to the wrong one.  SIFT's patent
    expired in 2020 and it is Apache-2.0 in main OpenCV, so there is no licence
    reason to avoid it.

    Content does not have to match exactly.  A different speed, a telltale that
    is off, a bar at another level: all measured within 0.01 px of the
    matched-content case, because RANSAC discards what moved and the rest of the
    screen carries the fit.
    """
    if display_size is None:
        display_size = (render.shape[1], render.shape[0])
    render_gray = to_gray(render)
    camera_gray = to_gray(camera_frame)

    sift = cv2.SIFT_create()
    kp_display, desc_display = sift.detectAndCompute(render_gray, None)
    kp_camera, desc_camera = sift.detectAndCompute(camera_gray, None)
    if desc_display is None or desc_camera is None or min(len(kp_display), len(kp_camera)) < 8:
        raise RuntimeError(
            "too little detail to match on. A nearly blank screen has nothing to "
            "align; put some content on it, or use a calibration pattern"
        )

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_display, desc_camera, k=2)
    # Lowe's ratio test: a keypoint whose best match is barely better than its
    # second best is ambiguous, and a cluster is full of near-identical tick
    # marks for it to be ambiguous between.
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if len(good) < min_inliers:
        raise RuntimeError(
            f"only {len(good)} confident matches between the framebuffer and the "
            "photograph. Check the render is of the screen actually on display, "
            "and that the whole screen is in frame"
        )

    src = np.float32([kp_display[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_camera[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("homography solve failed on the matched points")
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_inliers:
        raise RuntimeError(
            f"only {inliers} of {len(good)} matches agreed on a single mapping. "
            "That usually means the render and the photograph are of different "
            "screens"
        )

    method = "screen_content"
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(ecc_iterations), 1e-8)
        try:
            _, refined = cv2.findTransformECC(
                render_gray, camera_gray, H.astype(np.float32),
                cv2.MOTION_HOMOGRAPHY, criteria, None, 5,
            )
            H = np.asarray(refined, dtype=np.float64)
            method = "screen_content+ecc"
        except cv2.error:
            # The seed stands. It is the weaker answer and the caller is told.
            method = "screen_content(unrefined)"

    _check_display_quad(H, display_size, camera_frame.shape)
    residual = _homography_residual(
        src.reshape(-1, 2)[mask.ravel() == 1] if mask is not None else src.reshape(-1, 2),
        dst.reshape(-1, 2)[mask.ravel() == 1] if mask is not None else dst.reshape(-1, 2),
        H,
    )
    geometry = DisplayGeometry(
        H=H, display_size=display_size, method=method, residual_px=residual
    )
    geometry.inliers = inliers  # type: ignore[attr-defined]
    geometry.matches = len(good)  # type: ignore[attr-defined]
    return geometry


def _check_display_quad(
    H: np.ndarray, display_size: tuple[int, int], frame_shape: tuple[int, ...]
) -> None:
    """Refuse a mapping that does not put the display anywhere sensible.

    A homography fitted to bad correspondences can be arithmetically fine and
    geometrically nonsense -- folded over, inside out, or mapping the screen to a
    sliver.  None of that fails later in a way that points back here, so it is
    caught at the source.
    """
    w, h = display_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.isfinite(quad).all():
        raise RuntimeError("the solved mapping is not finite")
    if not cv2.isContourConvex(quad.astype(np.float32)):
        raise RuntimeError(
            "the solved mapping folds the display over itself, so the matches it "
            "was fitted to cannot all be right"
        )
    # Shoelace, signed: a convex quad stays convex when mirrored, so convexity
    # alone does not catch a mapping that has turned the screen back to front.
    # A camera looking at a display does not do that.
    signed = 0.0
    for i in range(4):
        x0, y0 = quad[i]
        x1, y1 = quad[(i + 1) % 4]
        signed += x0 * y1 - x1 * y0
    if signed <= 0:
        raise RuntimeError(
            "the solved mapping mirrors the display, which a camera looking at a "
            "screen does not do -- the matches it was fitted to are wrong"
        )
    area = abs(signed) / 2.0
    frame_area = float(frame_shape[0] * frame_shape[1])
    if not 0.01 * frame_area <= area <= 12.0 * frame_area:
        raise RuntimeError(
            f"the display maps to {area / frame_area:.3g} times the frame area, "
            "which is not a camera looking at a screen"
        )


# --------------------------------------------------------------------------
# ChArUco on the bezel
# --------------------------------------------------------------------------


@dataclass
class CharucoSpec:
    """The board stuck to the bezel, described well enough to detect it.

    ``square_length`` and ``marker_length`` are in whatever units the board was
    printed in; only their ratio and the board's proportions matter here,
    because the anchor below absorbs the scale.
    """

    squares_x: int
    squares_y: int
    square_length: float = 30.0
    marker_length: float = 22.0
    dictionary: str = "DICT_4X4_100"

    def board(self) -> "cv2.aruco.CharucoBoard":
        name = self.dictionary.upper()
        if not hasattr(cv2.aruco, name):
            raise ValueError(f"unknown ArUco dictionary {self.dictionary!r}")
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            float(self.square_length),
            float(self.marker_length),
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)),
        )

    @classmethod
    def parse(cls, text: str) -> "CharucoSpec":
        """``5x7``, ``5x7:30:22`` or ``5x7:30:22:DICT_4X4_100``."""
        parts = [p.strip() for p in str(text).split(":")]
        grid = parts[0].lower().replace("*", "x")
        if "x" not in grid:
            raise ValueError(f"expected COLSxROWS, got {parts[0]!r}")
        cols, _, rows = grid.partition("x")
        try:
            spec = cls(int(cols), int(rows))
        except ValueError as exc:
            raise ValueError(f"board size must be whole squares: {parts[0]!r}") from exc
        if spec.squares_x < 3 or spec.squares_y < 3:
            # Two rows of squares leave one row of inner corners, and a
            # homography cannot be fitted to collinear points.
            raise ValueError(
                f"a {spec.squares_x}x{spec.squares_y} board gives "
                f"{max(spec.squares_x - 1, 0)}x{max(spec.squares_y - 1, 0)} inner "
                "corners; at least 3x3 squares are needed for a usable one"
            )
        if len(parts) > 1 and parts[1]:
            spec.square_length = float(parts[1])
        if len(parts) > 2 and parts[2]:
            spec.marker_length = float(parts[2])
        else:
            spec.marker_length = spec.square_length * 0.75
        if len(parts) > 3 and parts[3]:
            spec.dictionary = parts[3]
        if not 0 < spec.marker_length < spec.square_length:
            raise ValueError("the marker has to be smaller than its square")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x, "squares_y": self.squares_y,
            "square_length": self.square_length, "marker_length": self.marker_length,
            "dictionary": self.dictionary,
        }


def detect_charuco(
    camera_img: np.ndarray, board: "cv2.aruco.CharucoBoard"
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolated chessboard corners of the bezel board, and their board ids.

    Marker corner refinement is **off**, straight from the OpenCV documentation:
    when the result feeds a homography, the proximity of the chessboard squares
    makes the sub-pixel step deviate, and those deviations propagate into the
    interpolated corners.
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)
    corners, ids, _, _ = detector.detectBoard(to_gray(camera_img))
    if corners is None or ids is None or len(corners) < 6:
        found = 0 if corners is None else len(corners)
        raise RuntimeError(
            f"only {found} board corners found. The board has to be fully in "
            "frame, in focus, and not washed out by glare"
        )
    return (
        np.asarray(corners, dtype=np.float64).reshape(-1, 2),
        np.asarray(ids).ravel(),
    )


def homography_board_to_camera(
    camera_img: np.ndarray, board: "cv2.aruco.CharucoBoard"
) -> np.ndarray:
    """Where the bezel board is, in the camera frame."""
    corners, ids = detect_charuco(camera_img, board)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
    H, _ = cv2.findHomography(board_points[ids], corners, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("could not fit a mapping to the board corners")
    return H


def charuco_anchor(
    camera_img: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
    geometry: DisplayGeometry,
) -> np.ndarray:
    """Learn where the active area sits relative to the bezel board.

    This is the one-time measurement that makes markers usable, and it needs a
    single frame in which *both* are visible: the board, and something that
    gives display-to-camera on its own -- the cluster's calibration screen, or
    its own content.  After this the markers alone are enough, for good, because
    a sticker on the bezel does not move when the software is reloaded.

    Returns the board-to-display transform, which is what
    :func:`homography_from_charuco` takes.
    """
    H_board_camera = homography_board_to_camera(camera_img, board)
    return np.linalg.inv(geometry.H) @ H_board_camera


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line fit.  Returns (point_on_line, unit_direction)."""
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


def _intersect(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-9:
        raise RuntimeError("display edges are parallel; cannot intersect for a corner")
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def refine_edges_subpixel(
    gray: np.ndarray,
    corners: np.ndarray,
    *,
    samples_per_edge: int = 120,
    half_width: float = 12.0,
) -> np.ndarray:
    """Pull a quadrilateral's edges onto the real intensity step.

    A threshold puts the boundary wherever the level happens to cut the blurred
    edge, which is a *consistent* place and therefore a bias rather than noise:
    on the simulated bezel it cost 0.86 px absolute while the scatter about it
    was only 0.31 px.  A bias that size is worth removing, and it is removable
    without knowing anything new -- the edge's true position is where the
    intensity is halfway between the two plateaus it separates, and that can be
    read off the profile to a fraction of a pixel.

    For each edge, intensity is sampled along the normal at many points, the
    half-height crossing is located by linear interpolation between the
    bracketing samples, and a line is fitted through those crossings.  Corners
    come from intersecting the refined lines, as before.
    """
    gray = gray.astype(np.float32)
    h, w = gray.shape[:2]
    refined_lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        along = b - a
        length = float(np.linalg.norm(along))
        if length < 4:
            raise RuntimeError("aperture edge too short to refine")
        along = along / length
        normal = np.array([-along[1], along[0]], np.float64)

        crossings = []
        for t in np.linspace(0.08, 0.92, samples_per_edge):
            centre = a + along * (length * t)
            offsets = np.arange(-half_width, half_width + 1.0, 1.0)
            pts = centre[None, :] + normal[None, :] * offsets[:, None]
            if (pts[:, 0].min() < 1 or pts[:, 0].max() > w - 2
                    or pts[:, 1].min() < 1 or pts[:, 1].max() > h - 2):
                continue
            profile = cv2.remap(
                gray, pts[:, 0].astype(np.float32).reshape(-1, 1),
                pts[:, 1].astype(np.float32).reshape(-1, 1),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
            # The plateaus are the ends of the profile; the step is between.
            near = float(np.median(profile[:4]))
            far = float(np.median(profile[-4:]))
            if abs(far - near) < 8.0:
                continue  # no real step here -- glare, or the edge is occluded
            half = 0.5 * (near + far)
            rising = far > near
            signed = profile - half
            if rising:
                idx = np.where((signed[:-1] < 0) & (signed[1:] >= 0))[0]
            else:
                idx = np.where((signed[:-1] > 0) & (signed[1:] <= 0))[0]
            if len(idx) != 1:
                continue  # ambiguous: more than one step along this normal
            k = int(idx[0])
            denom = profile[k + 1] - profile[k]
            frac = 0.0 if abs(denom) < 1e-6 else (half - profile[k]) / denom
            crossings.append(centre + normal * (offsets[k] + frac))

        if len(crossings) < 12:
            raise RuntimeError(
                f"aperture edge {i} gave only {len(crossings)} usable profiles; "
                "the boundary is not a clean step here -- check for glare, a "
                "reflection lying across it, or the trim and panel being the "
                "same brightness"
            )
        refined_lines.append(_fit_line(np.array(crossings)))

    # Edge i runs from corner i to corner i+1, so line i crossed with line i+1
    # is corner *i+1*. Roll by one to put each corner back at its own index --
    # the input order is already top-left first and must be preserved, since a
    # rotated quadrilateral fits a homography that is confidently wrong.
    crossed = np.array(
        [_intersect(*refined_lines[i], *refined_lines[(i + 1) % 4]) for i in range(4)],
        dtype=np.float64,
    )
    return np.roll(crossed, 1, axis=0)


def content_fills_aperture(
    frame: np.ndarray, corners: np.ndarray
) -> tuple[float, float, float]:
    """How much of a candidate aperture the screen's own content actually spans.

    The aspect-ratio test that finds the aperture cannot tell a display from
    anything else with the same proportions -- a laptop's own screen bezel
    around a windowed HMI, a reflection, a panel of trim.  When it picks the
    wrong one the homography is confidently wrong, the rectified frame is mostly
    empty, and the cluster ends up squeezed into a corner of display space.

    A display is showing something, and that something spans most of it.
    Measured on the simulated cluster: content spans 76% x 68% of the true
    active area, and of an aperture 1.6x too big only 48% x 42%, 2.5x too big
    31% x 27%.  So the span is a direct read on whether the rectangle found is
    the right size.

    Returns ``(width_span, height_span, lit_fraction)`` as fractions of the
    candidate.
    """
    w = int(round(max(np.linalg.norm(corners[1] - corners[0]),
                      np.linalg.norm(corners[2] - corners[3]))))
    h = int(round(max(np.linalg.norm(corners[3] - corners[0]),
                      np.linalg.norm(corners[2] - corners[1]))))
    if w < 8 or h < 8:
        return (0.0, 0.0, 0.0)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    H, _ = cv2.findHomography(corners, dst, method=0)
    if H is None:
        return (0.0, 0.0, 0.0)
    flat = cv2.warpPerspective(to_gray(frame), H, (w, h))

    # Content is what stands out from the screen's own background, whatever
    # that background happens to be -- a night theme is dark, a map screen is
    # not, so the threshold comes from the frame rather than from a constant.
    threshold = max(float(np.median(flat)) + 12.0, 40.0)
    ys, xs = np.where(flat > threshold)
    if len(xs) < 32:
        return (0.0, 0.0, 0.0)
    return (float(xs.max() - xs.min()) / w,
            float(ys.max() - ys.min()) / h,
            float(len(xs)) / flat.size)


def draw_aperture(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The frame with the detected border drawn on it, for looking at.

    When this route picks the wrong rectangle the number it produces is
    plausible, so the fastest way to see what happened is to see what it found.
    """
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    pts = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (0, 230, 255), 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(np.round(corners).astype(int)):
        cv2.circle(out, (int(x), int(y)), 9, (0, 230, 255), -1)
        cv2.putText(out, "TL TR BR BL".split()[i], (int(x) + 12, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2, cv2.LINE_AA)
    return out


def find_display_aperture(
    frame: np.ndarray,
    *,
    display_size: tuple[int, int],
    aspect_tolerance: float = 0.25,
    min_area_fraction: float = 0.02,
    max_area_fraction: float = 0.92,
    min_rectangularity: float = 0.80,
    search_width: int = 800,
    refine: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the display's physical opening in the bezel, whatever it is showing.

    This is the route for a cluster that will never draw anything for you.  It
    does not look for lit pixels, so it does not care what the HMI is showing:
    it looks for the *aperture* -- the boundary between the panel and the trim
    around it, which is a property of the hardware and is in every frame.

    **Why not just threshold.**  The obvious implementation is Otsu and take the
    big rectangle, and it does not work.  Otsu is a two-class split, and a
    photograph of a cluster has at least three populations in it: the dark
    screen, the mid-grey trim, and something bright -- a marker, a reflection, a
    lit telltale.  Measured on the simulated bezel, Otsu put its threshold at
    123, between the bright things and everything else, merging the screen and
    the trim into one blob covering 99.9% of the frame.  The display/bezel step
    was a clean 13-to-38 and the split went nowhere near it.

    So the threshold is *swept* instead.  At every level, both polarities are
    examined -- a dark screen in light trim and a lit screen in dark trim are
    both ordinary -- and every region that could be a display is scored.  The
    aperture is the region that stays the right shape across the widest range of
    levels, which is what makes it findable without knowing the brightnesses in
    advance.

    Contours are retrieved with ``RETR_LIST`` rather than ``RETR_EXTERNAL``,
    and that is not incidental: the aperture is a *hole*.  The trim surrounds
    it, so under any threshold that separates them the display is an interior
    boundary of the trim region, which ``RETR_EXTERNAL`` discards by definition
    -- and taken the other way round the dark screen merges with whatever dark
    scene surrounds the cluster and becomes one shapeless component.  Measured
    on the simulated bezel, that one flag was the difference between finding the
    aperture in every pose and finding it in none.

    Scoring uses the one thing known for free: the active area's **aspect
    ratio**.  A cluster frame is full of rectangles -- vents, trim, a binnacle,
    the reflection of a window -- and the display's is the one whose proportions
    match the framebuffer's.

    Returns the four sub-pixel corners and a dict of diagnostics, because when
    this picks the wrong rectangle you want to see why rather than guess.
    """
    gray = to_gray(frame)
    full_h, full_w = gray.shape[:2]
    want_aspect = display_size[0] / float(display_size[1])

    # Search small and fit big: the sweep is a shape search and does not need
    # the pixels, while the line fit that follows very much does.
    scale = min(1.0, search_width / float(full_w))
    small = cv2.GaussianBlur(
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray, (5, 5), 0)
    sh, sw = small.shape[:2]
    small_area = float(sh * sw)
    kernel = np.ones((5, 5), np.uint8)

    #: (level, polarity) -> the candidate's centre and size, so that a region
    #: surviving many consecutive levels can be recognised as the stable one.
    hits: list[dict[str, Any]] = []
    for level in range(8, 248, 3):
        _, binary = cv2.threshold(small, level, 255, cv2.THRESH_BINARY)
        for polarity, mask in (("bright", binary), ("dark", cv2.bitwise_not(binary))):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (min_area_fraction * small_area <= area
                        <= max_area_fraction * small_area):
                    continue
                (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour.astype(np.float32))
                if rw < 2 or rh < 2 or area / (rw * rh) < min_rectangularity:
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                if want_aspect < 1.0:
                    aspect = 1.0 / aspect
                err = abs(aspect - want_aspect) / want_aspect
                if err > aspect_tolerance:
                    continue
                hits.append({"level": level, "polarity": polarity, "centre": (cx, cy),
                             "size": (rw, rh), "aspect_error": err,
                             "rectangularity": area / (rw * rh),
                             "area_fraction": area / small_area})

    if not hits:
        raise RuntimeError(
            "no region in this frame has the display's proportions at any "
            f"threshold. The framebuffer is {display_size[0]}x{display_size[1]} "
            f"({want_aspect:.2f}:1), so that is what was looked for. The "
            "aperture has to be a visible boundary: get some light on the "
            "cluster so the trim and the panel do not read as one black shape, "
            "keep the whole display and a margin of bezel in frame, and watch "
            "for a reflection large enough to swallow an edge"
        )

    # Group hits that are the same region seen at different levels, then take
    # the group that survived the most levels -- stability is the signal. Ties
    # go to the better aspect match.
    groups: list[list[dict[str, Any]]] = []
    for hit in sorted(hits, key=lambda h: h["level"]):
        for group in groups:
            ref = group[-1]
            # Size is compared *relative to the candidate*, not as a fraction
            # of the frame. Absolute area fractions merged a display into the
            # screen around it -- 0.073 against 0.122 of frame is a difference
            # of 0.049, under any sane absolute tolerance, while being a
            # rectangle 30% wider. They then formed one group and the inner one
            # was never a separate candidate to prefer.
            ref_w = max(ref["size"])
            if (group[0]["polarity"] == hit["polarity"]
                    and abs(ref["centre"][0] - hit["centre"][0]) < 0.05 * sw
                    and abs(ref["centre"][1] - hit["centre"][1]) < 0.05 * sh
                    and abs(ref_w - max(hit["size"])) < 0.08 * ref_w):
                group.append(hit)
                break
        else:
            groups.append([hit])
    # Stability is the strongest signal, but it cannot stand alone: the trim's
    # own outline is just as stable as the aperture inside it, and on a cluster
    # whose trim has roughly the display's proportions it also passes the aspect
    # test. Measured at a sampling ratio of 3, the panel outline (aspect 2.30,
    # 60% of frame) was selected over the true aperture (aspect 2.65, 29%).
    # Rectangularity separates them -- an aperture fills its bounding rectangle
    # and a trim outline with a display cut out of it does not.
    def group_score(group: list[dict[str, Any]]) -> tuple[float, float]:
        levels = min(len(group) / 8.0, 1.0)
        rect = float(np.median([h["rectangularity"] for h in group]))
        aspect = 1.0 - min(float(np.median([h["aspect_error"] for h in group])) /
                           max(aspect_tolerance, 1e-6), 1.0)
        return (0.30 * levels + 0.40 * rect + 0.30 * aspect, rect)

    def group_width(group: list[dict[str, Any]]) -> float:
        return float(np.median([max(h["size"]) for h in group]))

    # Among the candidates that are credible on their own merits, take the
    # SMALLEST. Trim surrounds a display; a display never surrounds its trim, so
    # when several same-shaped rectangles are nested -- an HMI windowed on a
    # monitor, a display inside a binnacle inside a dash -- the innermost is the
    # active area and everything outside it is furniture.
    #
    # Picking the best-scoring one and then looking for something nested inside
    # it does not work: the outer rectangle scores at least as well as the inner
    # one on every term, so the search starts from the wrong place and the
    # nesting test has to be right about concentricity and relative size to
    # recover. Starting from the smallest credible candidate needs neither.
    scored = sorted(
        (g for g in groups if group_score(g)[0] >= 0.55 and len(g) >= 3),
        key=group_width,
    )
    if not scored:
        scored = [max(groups, key=group_score)]
    best = scored[0]
    if len(scored) > 1:
        # Only genuinely smaller candidates count as nested; near-duplicates of
        # the same rectangle are the same thing found twice.
        inner = group_width(best)
        nested = [g for g in scored[1:] if group_width(g) > 1.08 * inner]
        if nested:
            # Two credible rectangles of the display's shape, one inside the
            # other, and no reliable way to tell which is the active area.
            # "Innermost wins" is the right principle and it was not enough: on
            # a display windowed inside a screen of the same proportions it
            # picked correctly in 5 of 14 arrangements, and the other 9 were
            # silently wrong rather than refused -- a homography fitted to the
            # outer rectangle puts the whole cluster in a corner of display
            # space and every element then fails to match, which is what this
            # looked like in the field.
            #
            # Refusing is not a lesser answer here. The ambiguity is real, and
            # it is trivially removable by whoever is holding the camera.
            outer = group_width(nested[0])
            raise RuntimeError(
                f"two rectangles here have the display's proportions -- one "
                f"{inner:.0f} px wide and one {outer:.0f} px, one inside the "
                "other -- and which of them is the active area cannot be "
                "decided from the picture. That is usually an HMI running in a "
                "window, where the monitor's own border is the outer one. Run "
                "it full-screen (and pass that screen's resolution as the "
                "display size), or move in until the display and a thin margin "
                "of its surround fill the frame"
            )
    pick = min(best, key=lambda h: h["aspect_error"])

    # Now re-threshold at full resolution, at the level in the middle of the
    # stable range, and fit the edges there.
    levels = [h["level"] for h in best]
    level = int(round(float(np.median(levels))))
    _, binary = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0), level, 255, cv2.THRESH_BINARY)
    mask = binary if pick["polarity"] == "bright" else cv2.bitwise_not(binary)
    big = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, big)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    target = np.array(pick["centre"], np.float64) / max(scale, 1e-9)
    usable = [c for c in contours
              if cv2.contourArea(c) >= min_area_fraction * full_h * full_w]
    if not usable:
        raise RuntimeError(
            f"the aperture was found while searching but vanished at full "
            f"resolution (threshold {level}); this frame is probably too noisy "
            "or too unevenly lit for a single threshold to hold across it"
        )
    contour = min(
        usable,
        key=lambda c: float(np.linalg.norm(
            np.array(cv2.minAreaRect(c.astype(np.float32))[0]) - target)),
    )
    corners = corners_from_contour(contour.reshape(-1, 2).astype(np.float64))
    # An aperture running off the edge of the frame is the one failure worth
    # refusing outright. The visible part still fits a plausible rectangle, so
    # the homography comes back confident and wrong -- measured, a clipped
    # aperture at a sampling ratio of 2 gave 3.06 px while reporting nothing
    # unusual, against 0.05 px for the same shot uncropped. Every other failure
    # here raises; this one has to as well, or it is the only way this route
    # lies to you.
    margin = max(4.0, 0.004 * max(full_w, full_h))
    if (corners[:, 0].min() < margin or corners[:, 1].min() < margin
            or corners[:, 0].max() > full_w - 1 - margin
            or corners[:, 1].max() > full_h - 1 - margin):
        raise RuntimeError(
            "the display's boundary runs off the edge of the frame. The whole "
            "aperture and a margin of trim around it have to be visible, "
            "because the part in frame still fits a plausible rectangle and "
            "would be answered confidently. Stand back, or turn the camera"
        )
    # Is this rectangle the right *size*, not just the right shape? The aspect
    # test cannot tell the display from a laptop's own screen bezel around a
    # windowed HMI, or from any other rectangle of the same proportions, and
    # picking the wrong one produces a confident homography that puts the whole
    # cluster in a corner of display space.
    span_w, span_h, lit = content_fills_aperture(frame, corners)
    if lit > 0.0 and (span_w < 0.30 or span_h < 0.30):
        raise RuntimeError(
            f"found a {want_aspect:.2f}:1 rectangle, but what is lit inside it "
            f"spans only {span_w:.0%} by {span_h:.0%} of it -- so this is "
            "something larger than the active area, not the active area. The "
            "usual causes are the display size being wrong (its aspect ratio is "
            "what is searched for, so check --display-size against what the HMI "
            "is really running at) and, on a desk, the monitor's own bezel "
            "around a windowed HMI. A saved -aperture.jpg shows what was found"
        )
    refined = False
    if refine:
        # No silent fallback to the threshold corners. That fallback was written
        # first, on the reasoning that they are worth about a pixel and a
        # usable answer beats none -- and it was measured to be exactly wrong.
        # Whenever refinement failed, the coarse answer was 1.7 to 3.1 px out
        # while every refined one was inside 0.05 px, and nothing in the result
        # distinguished them. Refusing turns the one silent failure mode this
        # route had into a message.
        corners = refine_edges_subpixel(gray, corners)
        refined = True
    diagnostics = {
        "refined": refined,
        "content_span": (round(span_w, 3), round(span_h, 3)),
        "lit_fraction": round(lit, 4),
        "threshold": level,
        "polarity": pick["polarity"],
        "levels_stable": len(best),
        "aspect_error": pick["aspect_error"],
        "rectangularity": pick["rectangularity"],
        "score": group_score(best)[0],
        "area_fraction": pick["area_fraction"],
        "candidates": len(groups),
    }
    return corners, diagnostics


def homography_from_display_aperture(
    frame: np.ndarray,
    *,
    display_size: tuple[int, int],
    inset_px: tuple[float, float] = (0.0, 0.0),
) -> DisplayGeometry:
    """Method E -- the display's own physical border, with no cooperation at all.

    Methods A to D all ask the cluster for something: a pattern, a white frame,
    its framebuffer.  Even the bezel markers of method B need the active area
    located once, which means one calibration screen once.  This asks for
    nothing.  The aperture is hardware, it is in every frame, and it does not
    care what the software is doing.

    **What it costs, and why it is usually not a problem.**  The aperture is the
    physical opening; the active area is inset behind it by a mask whose width
    is a property of the module and is not visible from outside.  So this solves
    a frame that is offset and very slightly scaled against true display
    coordinates, by an amount this function cannot know.

    That constant does not matter for the measurement this package actually
    makes.  Reference and validate are both rectified through the *same*
    homography, so a fixed offset cancels exactly and a 6 px defect reads as
    6 px either way.  It matters only for comparing against a design in
    absolute display coordinates -- and there, ``inset_px`` takes the module's
    mask width if the datasheet or a ruler gives it to you.
    """
    corners_cam, diagnostics = find_display_aperture(
        frame, display_size=display_size)
    w, h = display_size
    ix, iy = float(inset_px[0]), float(inset_px[1])
    # Display coordinates of the aperture's corners: the active area grown by
    # the mask, since the opening is outside the pixels.
    # Half-pixel convention, and it is worth a measured 0.70 px if you get it
    # wrong. The aperture is the *outer* boundary of the edge pixels, and pixel
    # 0's outer boundary is at -0.5, not 0 -- so the opening spans -0.5 to
    # w-0.5, not 0 to w. Putting it at 0 to w offsets everything by half a pixel
    # on each axis, which is exactly the 0.707 px that showed up as a stubborn
    # constant bias before this line was written the right way round.
    corners_disp = np.array([
        [-0.5 - ix, -0.5 - iy],
        [w - 0.5 + ix, -0.5 - iy],
        [w - 0.5 + ix, h - 0.5 + iy],
        [-0.5 - ix, h - 0.5 + iy],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(corners_disp, corners_cam, method=0)
    if H is None:
        raise RuntimeError("homography solve failed")
    geometry = DisplayGeometry(
        H=H,
        display_size=display_size,
        method="display_aperture",
        residual_px=_homography_residual(corners_disp, corners_cam, H),
    )
    geometry.aperture = diagnostics
    return geometry


def corners_from_contour(
    contour: np.ndarray, *, min_edge_pixels: int = 20
) -> np.ndarray:
    """Four sub-pixel corners of a quadrilateral region, by fitting its edges.

    Lines are *fitted* to the edge pixels rather than corners being detected
    directly: a line fit averages over hundreds of edge pixels and lands well
    under a pixel, where a corner detector lands at about one.  Returned
    ordered top-left, top-right, bottom-right, bottom-left.
    """
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    quad = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32))).astype(np.float64)
    # Assign every contour pixel to its nearest quad edge, then fit each edge
    # over all of its pixels.
    edges: list[list[np.ndarray]] = [[] for _ in range(4)]
    for pt in contour:
        best, best_d = 0, float("inf")
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            ab = b - a
            t = np.clip(np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
            d = np.linalg.norm(pt - (a + t * ab))
            if d < best_d:
                best, best_d = i, d
        edges[best].append(pt)

    fits = []
    for i, pts in enumerate(edges):
        if len(pts) < min_edge_pixels:
            raise RuntimeError(f"display edge {i} had only {len(pts)} pixels to fit")
        fits.append(_fit_line(np.array(pts)))

    corners = np.array(
        [_intersect(*fits[i], *fits[(i + 1) % 4]) for i in range(4)], dtype=np.float64
    )
    centre = corners.mean(axis=0)
    order = np.argsort(np.arctan2(*(corners - centre).T[::-1]))
    corners = corners[order]
    start = int(np.argmin(np.sum((corners - centre) * [[1, 1]], axis=1)))
    return np.roll(corners, -start, axis=0)


def homography_from_display_edges(
    white_frame: np.ndarray,
    *,
    display_size: tuple[int, int],
    threshold: int | None = None,
) -> DisplayGeometry:
    """Method C -- find the display's own edges on a full-white screen.

    Lines are *fitted* to the edge pixels rather than corners being detected
    directly: a line fit averages over hundreds of edge pixels and lands well
    under a pixel, where a corner detector lands at about one.
    """
    gray = to_gray(white_frame)
    if threshold is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no lit region found in the full-white frame")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

    corners_cam = corners_from_contour(contour)

    w, h = display_size
    corners_disp = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    H, _ = cv2.findHomography(corners_disp, corners_cam, method=0)
    if H is None:
        raise RuntimeError("homography solve failed")
    return DisplayGeometry(
        H=H,
        display_size=display_size,
        method="display_edges",
        residual_px=_homography_residual(corners_disp, corners_cam, H),
    )


# --------------------------------------------------------------------------
# per-frame drift correction
# --------------------------------------------------------------------------


@dataclass
class DriftEstimate:
    magnitude_px: float
    rotation_deg: float
    correlation: float
    converged: bool
    warp_camera: np.ndarray
    """3x3 camera-space warp taking reference-camera coordinates to live ones."""

    @property
    def exceeds(self) -> bool:  # set by DriftTracker.measure
        return getattr(self, "_exceeds", False)


class DriftTracker:
    """Per-frame pose correction against a region that never changes.

    Rigs get bumped.  The correction itself is cheap; the important part is that
    it is **loud**.  Silent compensation is how a rig that someone knocked last
    Tuesday keeps producing green results for a month.

    ``motion`` picks how much of a pose change is modelled:

    ``cv2.MOTION_EUCLIDEAN`` (the default)
        Rotation and translation.  Right for a mounted camera that has been
        nudged: it cannot absorb much, so anything larger shows up as a failure
        to converge rather than as a silent correction.
    ``cv2.MOTION_HOMOGRAPHY``
        The full pose.  What a hand-held camera needs, because moving it changes
        perspective and not just position.

    There is a real cost to the homography mode, and it is not a detail: a
    correction that re-solves the whole pose will also absorb a defect in which
    *everything* moved together, and report nothing.  Per-element faults still
    show up, because the rest of the frame dominates the fit.  A whole-layout
    shift does not.  Use Euclidean on a mounted rig, where that trade is not
    needed.
    """

    def __init__(
        self,
        reference_camera_frame: np.ndarray,
        static_roi: tuple[int, int, int, int],
        *,
        alarm_px: float = 2.0,
        max_iterations: int = 200,
        eps: float = 1e-6,
        gauss_filt_size: int = 5,
        motion: int = cv2.MOTION_EUCLIDEAN,
    ) -> None:
        x, y, w, h = static_roi
        self.roi = (int(x), int(y), int(w), int(h))
        self.alarm_px = float(alarm_px)
        self.motion = int(motion)
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(max_iterations),
            float(eps),
        )
        self._gauss = int(gauss_filt_size)
        self.reference = self._crop(reference_camera_frame)

    def _crop(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return to_gray(img)[y : y + h, x : x + w].astype(np.float32)

    def measure(self, live_camera_frame: np.ndarray) -> DriftEstimate:
        live = self._crop(live_camera_frame)
        homography = self.motion == cv2.MOTION_HOMOGRAPHY
        warp = (np.eye(3, 3, dtype=np.float32) if homography
                else np.eye(2, 3, dtype=np.float32))
        try:
            cc, warp = cv2.findTransformECC(
                self.reference,
                live,
                warp,
                self.motion,
                self._criteria,
                None,
                self._gauss,
            )
            converged = True
        except cv2.error:
            # ECC throws when it cannot converge.  A non-converging drift
            # estimate is itself a finding -- the static region no longer looks
            # like the static region.
            cc, converged = 0.0, False

        # ECC works in ROI-local coordinates; conjugate by the ROI offset to get
        # a warp valid over the whole camera frame.
        x, y, _, _ = self.roi
        T = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float64)
        warp = np.asarray(warp, np.float64)
        W_local = warp if homography else np.vstack([warp, [0, 0, 1]])
        W_full = T @ W_local @ np.linalg.inv(T)

        est = DriftEstimate(
            magnitude_px=float(np.hypot(warp[0, 2], warp[1, 2])),
            rotation_deg=float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))),
            correlation=float(cc),
            converged=converged,
            warp_camera=W_full,
        )
        est._exceeds = (not converged) or est.magnitude_px > self.alarm_px  # type: ignore[attr-defined]
        return est

    def corrected_homography(self, geometry: DisplayGeometry, est: DriftEstimate) -> np.ndarray:
        """Base homography with the measured drift applied on top.

        ``H`` maps display -> reference camera; the drift warp maps reference
        camera -> live camera; so the live mapping is their composition.
        """
        if not est.converged:
            return geometry.H
        return est.warp_camera @ geometry.H

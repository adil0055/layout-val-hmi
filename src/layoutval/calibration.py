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
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # see docstring
    detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)

    gray = to_gray(camera_img)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < 6:
        raise RuntimeError(
            f"ChArUco: only {0 if charuco_corners is None else len(charuco_corners)} "
            "interpolated corners found; need at least 6"
        )

    board_pts = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
    ids = np.asarray(charuco_ids).ravel()
    src_board = board_pts[ids]
    cam = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)

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
        if len(pts) < 20:
            raise RuntimeError(f"display edge {i} had only {len(pts)} pixels to fit")
        fits.append(_fit_line(np.array(pts)))

    corners_cam = np.array(
        [_intersect(*fits[i], *fits[(i + 1) % 4]) for i in range(4)], dtype=np.float64
    )
    # Order: top-left, top-right, bottom-right, bottom-left.
    centre = corners_cam.mean(axis=0)
    order = np.argsort(np.arctan2(*(corners_cam - centre).T[::-1]))
    corners_cam = corners_cam[order]
    start = int(np.argmin(np.sum((corners_cam - centre) * [[1, 1]], axis=1)))
    corners_cam = np.roll(corners_cam, -start, axis=0)

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

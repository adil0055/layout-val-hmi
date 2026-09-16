# Licences, checked

This domain has one genuinely expensive trap and several smaller ones. The table
below is what this project depends on and what it deliberately avoids.

`tests/test_architecture.py::test_runtime_dependencies_are_all_permissively_licensed`
fails the build if one of the blocked names appears in `pyproject.toml`.

## What this project uses

| Component | Licence | Commercial | Note |
|---|---|---|---|
| OpenCV (main modules) | Apache-2.0 | Clear | Includes ArUco/ChArUco since 4.7 |
| opencv-contrib | Apache-2.0 | Check build | Build with `OPENCV_ENABLE_NONFREE=OFF`; SURF sits behind that flag |
| NumPy, SciPy | BSD-3 | Clear | |
| scikit-image | BSD-3 | Clear | SSIM lives here |
| PyYAML | MIT | Clear | |

`layoutval` itself is Apache-2.0.

## What is safe if you need it

| Component | Licence | Commercial | Note |
|---|---|---|---|
| AprilTag (UMich) | BSD-2 | Clear | Alternative to ArUco |
| SIFT | Apache-2.0 | Clear | Patent expired 2020; in main OpenCV |
| SAM / SAM 2 | Apache-2.0 | Clear | Verify SAM 3 separately — newer release, different terms |
| Grounding DINO / Grounded-SAM | Apache-2.0 | Clear | Zero-shot boxes from a text prompt, for authoring |
| Tesseract, PaddleOCR, EasyOCR | Apache-2.0 | Clear | PaddleOCR is stronger on screen text |
| YOLOX, MMDetection, Detectron2 | Apache-2.0 | Clear | If a detector is ever genuinely required |
| PyTorch | BSD-3 | Clear | |
| LightGlue (code + weights) | Apache-2.0 | Partial | Pair with DISK, ALIKED or SIFT — **not** SuperPoint |

## What to avoid

| Component | Licence | Commercial | Note |
|---|---|---|---|
| SuperPoint | Non-commercial | Blocked | Restrictive licence covers the weights **and** the inference file |
| Surya OCR | GPL-3.0 / OpenRAIL-M | Blocked | Code is GPL, weights carry a modified OpenRAIL |
| Ultralytics YOLO (v5 – current) | AGPL-3.0 | Licence needed | Applies to internal use and to fine-tuned weights |
| SURF | Patent-encumbered | Avoid | |

## The two worth spelling out

**Ultralytics YOLO** ships under AGPL-3.0. Their own position is that using the
code, architectures, training pipelines *or trained weights* requires either
open-sourcing your entire project under AGPL-3.0 or buying an Enterprise licence,
and they state explicitly that internal company use — including internal R&D —
needs the Enterprise licence. For a validation tool used on client programmes
that is a legal review and a recurring line item, not a footnote. If a detector
is genuinely needed somewhere, YOLOX, RT-DETR via PaddleDetection, MMDetection and
Detectron2 are Apache-2.0 and carry none of it.

**SuperPoint** is the trap worth naming, because it is the default pairing in
nearly every LightGlue tutorial. LightGlue's own repository is explicit that its
code and weights are Apache-2.0, and DISK follows the same, but SuperPoint carries
a different, restrictive licence covering both its pre-trained weights and its
inference file. Copy a tutorial without reading it and you have imported a
non-commercial dependency into a commercial product.

## Where a model is allowed at all

`layoutval.authoring` is the only module that may touch one, and it declares the
backend as a `Protocol` so the package never depends on a model at import time.
`tests/test_architecture.py` asserts that no module on the measurement path
imports torch, ultralytics, segment-anything or `layoutval.authoring` itself.

The line is: **learned models in the authoring loop, deterministic code in the
measurement loop.** Not an anti-ML position — it puts ML where its failure mode is
"an engineer adjusts a box" rather than "a safety telltale defect ships". SAM and
SAM 2 are Apache-2.0, run offline, and a human confirms every output before
`accept_proposal` will turn it into an inventory element.

## References

- OpenCV — [ChArUco board detection](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html), including the corner-refinement caveat for homography use
- [LightGlue repository](https://github.com/cvg/lightglue) — licence split between LightGlue/DISK and SuperPoint
- [Ultralytics licensing](https://www.ultralytics.com/license) and [their statement on internal company use](https://github.com/orgs/ultralytics/discussions/1260)

#!/usr/bin/env python3
"""SAM 3 segmentation service for layoutval -- runs on the computer with the GPU.

layoutval (on the bench computer) sends a photograph and a text prompt; this
answers with the masks SAM 3 finds for that prompt. Nothing else is shared:
the two computers only talk over HTTP, so layoutval never needs PyTorch.

    POST /segment?prompt=<text>&threshold=<0..1>&max=<n>
        body:   the photograph, JPEG or PNG bytes
        header: X-Token: <token>          (when started with --token)
        200 ->  {"width": W, "height": H, "prompt": "...", "seconds": t,
                 "results": [{"score": s, "box": [x0, y0, x1, y1], "area": px,
                              "mask_png": "<base64 PNG, 0/255, W x H>"}, ...]}
                best score first; "results" is empty when nothing matched.

    GET /health -> {"ok": true, "model": "sam3", "device": "cuda", ...}

Run:
    python sam3_server.py --port 8765 --token <shared secret>
then start layoutval with
    layoutval go --sam3-url http://<this computer>:8765 --sam3-token <shared secret>

Needs the official SAM 3 package and its checkpoint (gated on Hugging Face:
request access at huggingface.co/facebook/sam3, then `hf auth login`), or a
local checkpoint file with --checkpoint. See README.md beside this file.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

#: Biggest photograph accepted, bytes.
MAX_UPLOAD = 60 * 1024 * 1024
#: Masks scoring below this are not returned, unless the request asks otherwise.
DEFAULT_THRESHOLD = 0.4
#: Most masks returned per request.
DEFAULT_MAX = 5


def _skip_clone_beside_script() -> None:
    """Keep a SAM 3 clone that sits beside this file from shadowing the installed package.

    Saved in the folder the repository was cloned into, this file has a folder
    named ``sam3`` next to it -- the clone, which has no ``__init__.py`` -- and
    Python looks in this file's folder first. ``sam3`` was then imported as an
    empty namespace package, its submodules still found through the editable
    install, and SAM 3 failed loading its tokenizer: "expected str, bytes or
    os.PathLike object, not NoneType", from pkg_resources.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    clone = os.path.join(here, "sam3")
    if os.path.isdir(clone) and not os.path.isfile(os.path.join(clone, "__init__.py")):
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]


class Sam3Segmenter:
    """SAM 3 itself: one model, loaded once, asked one photograph at a time."""

    def __init__(self, device: str = "auto", checkpoint: str | None = None,
                 bf16: bool = True) -> None:
        # Imported here, not at the top, so the HTTP part of this file can be
        # tested on a computer without PyTorch.
        import torch
        _skip_clone_beside_script()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print(f"  no usable GPU: PyTorch {torch.__version__} (CUDA "
                      f"{torch.version.cuda or 'none'}) cannot use one here, so SAM 3 "
                      "runs on the CPU, many times slower. Check nvidia-smi.", flush=True)
        self.device = device
        if device == "cuda":
            # TF32 on Ampere and later: faster, and nothing here needs more.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.bf16 = bool(bf16 and device == "cuda" and torch.cuda.is_bf16_supported())
        kwargs = {"device": device}
        if checkpoint:
            kwargs.update(checkpoint_path=checkpoint, load_from_HF=False)
        self.model = build_sam3_image_model(**kwargs)
        self.processor = Sam3Processor(self.model, device=device,
                                       confidence_threshold=DEFAULT_THRESHOLD)
        self.lock = threading.Lock()

    def segment(self, image: Image.Image, prompt: str, threshold: float,
                limit: int) -> list[dict]:
        torch = self.torch
        with self.lock, torch.inference_mode():
            # Applied inside the model: below it no full-size mask is even made
            # (there are a couple of hundred candidates per photograph).
            self.processor.confidence_threshold = threshold
            if self.bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self._run(image, prompt)
            else:
                out = self._run(image, prompt)
        masks = out.get("masks")
        if masks is None or len(masks) == 0:
            return []
        masks = masks.detach().to("cpu").numpy().astype(bool)
        if masks.ndim == 4:                           # [N, 1, H, W]
            masks = masks[:, 0]
        scores = out["scores"].detach().float().to("cpu").numpy().reshape(-1)
        boxes = out["boxes"].detach().float().to("cpu").numpy().reshape(-1, 4)
        found = []
        for mask, score, box in zip(masks, scores, boxes):
            if float(score) < threshold or not mask.any():
                continue
            found.append({"mask": mask, "score": float(score),
                          "box": [float(v) for v in box]})
        found.sort(key=lambda r: r["score"], reverse=True)
        return found[:limit]

    def _run(self, image: Image.Image, prompt: str) -> dict:
        state = self.processor.set_image(image)
        return self.processor.set_text_prompt(prompt=prompt, state=state)


def encode_mask(mask: np.ndarray) -> str:
    """A boolean mask as a base64 PNG, 0 and 255."""
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    server_version = "layoutval-sam3/1"

    def log_message(self, fmt, *args):  # one line per request, from _done
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if not token:
            return True
        given = self.headers.get("X-Token", "")
        return hmac.compare_digest(given.encode(), token.encode())

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        seg = self.server.segmenter  # type: ignore[attr-defined]
        self._json(HTTPStatus.OK, {"ok": True, "model": "sam3",
                                   "device": getattr(seg, "device", "?"),
                                   "auth": bool(self.server.token)})  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        started = time.monotonic()
        url = urlparse(self.path)
        if url.path != "/segment":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorised():
            self._json(HTTPStatus.FORBIDDEN, {"error": "bad or missing X-Token"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                       {"error": f"send a photograph of 1 byte to {MAX_UPLOAD // 2**20} MB"})
            return
        query = parse_qs(url.query)
        prompt = (query.get("prompt", [""])[0] or "").strip()
        if not prompt:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "prompt is empty"})
            return
        try:
            threshold = min(0.99, max(0.05, float(query.get("threshold", [DEFAULT_THRESHOLD])[0])))
            limit = max(1, min(20, int(query.get("max", [DEFAULT_MAX])[0])))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "threshold and max are numbers"})
            return
        try:
            image = Image.open(io.BytesIO(self.rfile.read(length)))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - anything unreadable
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"not an image: {exc}"})
            return
        try:
            found = self.server.segmenter.segment(  # type: ignore[attr-defined]
                image, prompt, threshold, limit)
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"{type(exc).__name__}: {exc}"})
            print(f"  segment failed: {type(exc).__name__}: {exc}", flush=True)
            return
        w, h = image.size
        results = []
        for r in found:
            mask = np.asarray(r["mask"], dtype=bool)
            if mask.shape != (h, w):
                continue                                  # never send a mask of another size
            results.append({"score": round(r["score"], 4),
                            "box": [round(v, 1) for v in r["box"]],
                            "area": int(mask.sum()),
                            "mask_png": encode_mask(mask)})
        seconds = round(time.monotonic() - started, 2)
        self._json(HTTPStatus.OK, {"width": w, "height": h, "prompt": prompt,
                                   "seconds": seconds, "results": results})
        best = f", best {results[0]['score']:.2f}" if results else ""
        print(f"  {self.client_address[0]} '{prompt}' {w}x{h}: "
              f"{len(results)} mask(s){best} in {seconds}s", flush=True)


class Sam3Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, segmenter, token: str = "") -> None:
        super().__init__(address, Handler)
        self.segmenter = segmenter
        self.token = token


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SAM 3 segmentation service for layoutval")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("SAM3_TOKEN", ""),
                    help="shared secret layoutval must send (or set SAM3_TOKEN); "
                         "empty means anyone on the network may use it")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--checkpoint", default=None,
                    help="a local sam3.pt instead of downloading from Hugging Face")
    ap.add_argument("--fp32", action="store_true",
                    help="no bfloat16 autocast on the GPU (slower, for older cards)")
    args = ap.parse_args(argv)

    print("loading SAM 3 ...", flush=True)
    started = time.monotonic()
    segmenter = Sam3Segmenter(device=args.device, checkpoint=args.checkpoint,
                              bf16=not args.fp32)
    print(f"SAM 3 ready on {segmenter.device} in {time.monotonic() - started:.0f}s", flush=True)
    server = Sam3Server((args.host, args.port), segmenter, token=args.token)
    print(f"listening on http://{args.host}:{args.port}  "
          f"(token {'required' if args.token else 'not required'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

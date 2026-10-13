"""Every element on its own, against its name: pictures for a person to look through.

A run's overlay shows the whole screen with a box and a label on each
element, and at a cluster's density the labels sit on top of each other. So
each element is also cut out on its own -- as the reference has it and as the
run saw it, its box drawn round it -- and put against its name and its
verdict: one PNG per element in ``elements/``, and all of them on one page,
``elements.html``, that needs nothing but a browser.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: Context round an element: this share of its size, at least PAD_MIN px.
PAD = 0.35
PAD_MIN = 6
#: Pictures are scaled up to at least this high, and no wider than this.
MIN_H = 72
MAX_W = 320

COLOURS = {"PASS": (132, 181, 79), "REVIEW": (63, 168, 220), "FAIL": (92, 105, 232)}


def crop(img: np.ndarray, box, pad: float = PAD) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """``box`` (x, y, w, h) with some context round it; and where the box is in the crop."""
    x, y, w, h = (int(round(v)) for v in box)
    p = max(PAD_MIN, int(round(pad * max(w, h))))
    H, W = img.shape[:2]
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(W, x + w + p), min(H, y + h + p)
    out = img[y0:y1, x0:x1]
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out.copy(), (x - x0, y - y0, w, h)


def picture(img: np.ndarray, box, colour=(200, 200, 200)) -> np.ndarray:
    """An element cut out, scaled to be seen, its box drawn round it."""
    out, (bx, by, bw, bh) = crop(img, box)
    h, w = out.shape[:2]
    k = max(1.0, MIN_H / max(h, 1))
    k = min(k, MAX_W / max(w, 1)) if w * k > MAX_W else k
    if abs(k - 1.0) > 1e-3:
        out = cv2.resize(out, (max(1, round(w * k)), max(1, round(h * k))),
                         interpolation=cv2.INTER_NEAREST if k >= 2 else cv2.INTER_AREA)
    cv2.rectangle(out, (int(bx * k) - 1, int(by * k) - 1),
                  (int((bx + bw) * k), int((by + bh) * k)), colour, 1)
    return out


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "element"


def card(item: dict[str, Any]) -> np.ndarray:
    """One element as a PNG: its name and verdict over its pictures, labelled."""
    pics = item["pictures"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    gap, label_h, head_h = 8, 18, 40
    width = max(320, sum(p.shape[1] for _, p in pics) + gap * (len(pics) + 1))
    height = head_h + max(p.shape[0] for _, p in pics) + label_h + gap
    out = np.full((height, width, 3), (24, 21, 15), np.uint8)
    colour = COLOURS.get(item.get("verdict", ""), (200, 200, 200))
    cv2.putText(out, _ascii(item["name"])[:48], (gap, 17), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, _ascii(f"{item.get('verdict', '')}  {item.get('detail', '')}")[:56],
                (gap, 34), font, 0.42, colour, 1, cv2.LINE_AA)
    x = gap
    for label, p in pics:
        h, w = p.shape[:2]
        out[head_h:head_h + h, x:x + w] = p
        cv2.putText(out, label, (x, head_h + h + 13), font, 0.4, (160, 170, 175), 1, cv2.LINE_AA)
        x += w + gap
    return out


def _fit(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img if w <= width else cv2.resize(img, (width, max(1, round(h * width / w))),
                                              interpolation=cv2.INTER_AREA)


def _data_url(img: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii") if ok else ""


def write(folder: Path, items: list[dict[str, Any]], title: str,
          problems: list[dict[str, Any]] = (),
          views: list[tuple[str, np.ndarray]] = ()) -> Path:
    """``elements/NNN-name.png`` for each item, and ``elements.html`` with them all.

    An item is {"name", "kind" ("lamp" or "element"), "verdict", "detail",
    "pictures": [(label, image), ...]}. ``problems`` are the run's rows that
    did not pass ({"id", "verdict", "detail"}) and ``views`` whole pictures
    that show them (the bulb check, the lamp check): they head the page, so
    that what it shows agrees with the verdict -- a bulb check that saw no
    lamp is no element's, and the page was all green under a FAIL.
    """
    pics_dir = folder / "elements"
    pics_dir.mkdir(parents=True, exist_ok=True)
    for old in pics_dir.glob("*.png"):
        old.unlink()
    sections: dict[str, list[str]] = {"lamp": [], "named": [], "unknown": []}
    order = {"FAIL": 0, "REVIEW": 1}
    items = sorted(items, key=lambda it: order.get(it.get("verdict", ""), 2))
    for i, item in enumerate(items, 1):
        cv2.imwrite(str(pics_dir / f"{i:03d}-{_slug(item['name'])}.png"), card(item))
        kind = item.get("kind", "element")
        if kind != "lamp":
            kind = "unknown" if item["name"].startswith(("unknown", "auto@")) else "named"
        imgs = "".join(
            f'<figure><img src="{_data_url(p)}" alt=""><figcaption>{html.escape(label)}</figcaption>'
            f"</figure>" for label, p in item["pictures"])
        v = html.escape(item.get("verdict", ""))
        sections[kind].append(
            f'<div class="card"><div class="name">{html.escape(item["name"])}</div>'
            f'<div class="v {v}">{v}<span>{html.escape(item.get("detail", ""))}</span></div>'
            f'<div class="pics">{imgs}</div></div>')
    heads = {"lamp": "Lamps", "named": "Named elements", "unknown": "Other elements"}
    body = "".join(
        f"<h2>{heads[k]} <small>{len(v)}</small></h2><div class=grid>{''.join(v)}</div>"
        for k, v in sections.items() if v)
    if problems or views:
        rows = "".join(
            f'<tr><td class="{html.escape(r.get("verdict", ""))}">{html.escape(str(r.get("id", "")))}'
            f'</td><td>{html.escape(str(r.get("detail", "")))}</td></tr>' for r in problems)
        shots = "".join(
            f'<figure><img class="view" src="{_data_url(_fit(img, 900))}" alt="">'
            f'<figcaption>{html.escape(label)}</figcaption></figure>' for label, img in views)
        body = (f"<h2>Not passing <small>{len(problems)}</small></h2>"
                + (f"<table>{rows}</table>" if rows else "")
                + (f'<div class="pics">{shots}</div>' if shots else "") + body)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{margin:0;padding:16px;background:#0F1518;color:#E3EAE8;
     font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px}}
h2 small{{color:#8B9AA3;font-weight:400}} p{{color:#8B9AA3;margin:0 0 12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}}
.card{{background:#171F23;border:1px solid #2B373D;border-radius:9px;padding:10px;min-width:0}}
.name{{font-weight:600;overflow-wrap:anywhere}}
.v{{font-size:12px;font-weight:700;margin:2px 0 8px}} .v span{{color:#9DADB5;font-weight:400;margin-left:6px}}
.PASS{{color:#4FB584}} .REVIEW{{color:#DCA83F}} .FAIL{{color:#E8695C}}
.pics{{display:flex;gap:8px;flex-wrap:wrap}} figure{{margin:0}}
figure img{{display:block;max-width:100%;image-rendering:pixelated;border-radius:4px;background:#000}}
figcaption{{color:#8B9AA3;font-size:11px}}
table{{border-collapse:collapse;margin:0 0 12px;max-width:100%}}
td{{padding:4px 10px 4px 0;border-bottom:1px solid #222C31;vertical-align:top;overflow-wrap:anywhere}}
img.view{{image-rendering:auto;max-width:min(900px,100%)}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Each element cut out on its own, its box drawn round it: as the reference has it and as
this run saw it. Failures and reviews first.</p>
{body}
</body></html>"""
    target = folder / "elements.html"
    target.write_text(page, encoding="utf-8")
    return target

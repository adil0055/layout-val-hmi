"""The element inventory: ``layout_profile.yaml`` and its reference patches.

The inventory is established once, during authoring, and at runtime the pipeline
only measures.  Three sources, ranked, recorded per element in
:attr:`ElementSpec.source`:

``design``
    Exported from the HMI design tool (Kanzi, Altia, CGI Studio, Qt all hold a
    node tree with position and size for every element).  Expected positions
    become traceable to a design artefact rather than to a photograph somebody
    took once, and the test answers "does the build match the design" instead of
    "does the build match the last build".  It also survives a rig change
    untouched, which a photograph does not.
``teachin``
    Differential teach-in against a CAN signal -- see :mod:`layoutval.teachin`.
``manual``
    Assisted annotation, optionally seeded by a segmentation model with an
    engineer confirming every box -- see :mod:`layoutval.authoring`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np
import yaml

from layoutval.types import AngleModel, ElementKind, ElementSpec, PositionModel, Tolerance


def _tuple4(v: Iterable[float]) -> tuple[float, float, float, float]:
    a, b, c, d = (float(x) for x in v)
    return a, b, c, d


class LayoutProfile:
    """One screen, one theme, and everything expected to be on it.

    A profile is per *theme* on purpose.  The same element at 20% backlight has a
    different edge profile, and while ZNCC absorbs a global gain and offset
    change it does not handle a gamma change well.  Teach day and night
    separately rather than hoping one reference generalises.
    """

    def __init__(
        self,
        screen: str,
        display_size: tuple[int, int],
        *,
        theme: str | None = None,
        elements: list[ElementSpec] | None = None,
        defaults: Tolerance | None = None,
        reference_path: str | None = None,
        reference_values: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> None:
        self.screen = screen
        self.display_size = (int(display_size[0]), int(display_size[1]))
        self.theme = theme
        self.elements = elements or []
        self.defaults = defaults or Tolerance()
        self.reference_path = reference_path
        self.reference_values = reference_values or {}
        """Signal values the reference frame was captured at.  Anything that
        moves is only comparable to the reference at these values."""

        self.metadata = metadata or {}
        self.root = Path(root) if root else Path(".")
        self._templates: dict[str, np.ndarray] = {}

    # -- container behaviour ------------------------------------------------

    def __iter__(self) -> Iterator[ElementSpec]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, element_id: str) -> ElementSpec:
        for e in self.elements:
            if e.id == element_id:
                return e
        raise KeyError(element_id)

    def add(self, spec: ElementSpec) -> None:
        if any(e.id == spec.id for e in self.elements):
            raise ValueError(f"duplicate element id {spec.id!r}")
        self.elements.append(spec)

    # -- reference imagery --------------------------------------------------

    def reference(self) -> np.ndarray:
        """The rectified golden frame for this screen and theme."""
        if not self.reference_path:
            raise RuntimeError(f"profile {self.screen!r} has no reference image")
        path = self.root / self.reference_path
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read reference image {path}")
        return img

    def template(self, element_id: str) -> np.ndarray | None:
        """The stored reference patch for an element, if it has one."""
        if element_id in self._templates:
            return self._templates[element_id]
        spec = self[element_id]
        if not spec.template_path:
            return None
        path = self.root / spec.template_path
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read template {path} for {element_id!r}")
        self._templates[element_id] = img
        return img

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen": self.screen,
            "theme": self.theme,
            "display_size": list(self.display_size),
            "reference": self.reference_path,
            "reference_values": self.reference_values,
            "metadata": self.metadata,
            "defaults": {"tolerance": asdict(self.defaults)},
            "elements": [_spec_to_dict(e, self.defaults) for e in self.elements],
        }

    def save(
        self,
        path: Path | str,
        *,
        templates: Mapping[str, np.ndarray] | None = None,
        anchors: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        """Write the profile, and any in-memory templates, next to it.

        ``anchors`` gives the display coordinate each template's top-left was cut
        at.  Pass it whenever the template does not come from the element's own
        bbox -- a teach-in template cut to the ink of a design-sourced element,
        for instance.  Without it the padding between the two is measured as a
        displacement on every run.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        root = path.parent
        if templates:
            tdir = root / "templates"
            tdir.mkdir(exist_ok=True)
            for eid, patch in templates.items():
                rel = f"templates/{_safe_name(eid)}.png"
                cv2.imwrite(str(root / rel), patch)
                self[eid].template_path = rel
                if anchors and eid in anchors:
                    self[eid].template_anchor = (
                        float(anchors[eid][0]),
                        float(anchors[eid][1]),
                    )
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        self.root = root

    @classmethod
    def load(cls, path: Path | str) -> "LayoutProfile":
        path = Path(path)
        d = yaml.safe_load(path.read_text())
        defaults = Tolerance(**(d.get("defaults", {}).get("tolerance", {}) or {}))
        elements = [_spec_from_dict(e, defaults) for e in d.get("elements", [])]
        return cls(
            screen=d["screen"],
            display_size=tuple(d["display_size"]),
            theme=d.get("theme"),
            elements=elements,
            defaults=defaults,
            reference_path=d.get("reference"),
            reference_values=d.get("reference_values") or {},
            metadata=d.get("metadata", {}),
            root=path.parent,
        )

    # -- self-checks --------------------------------------------------------

    def validate(self) -> list[str]:
        """Problems worth refusing to run on.

        Includes the tolerance-versus-noise check: a ``tol_fail`` tighter than
        ``3*sigma + spec`` describes a test the rig cannot actually perform, and
        running it anyway produces a suite that fails randomly, gets marked flaky
        and is ignored inside two months.
        """
        problems: list[str] = []
        seen: set[str] = set()
        w, h = self.display_size
        for e in self.elements:
            if e.id in seen:
                problems.append(f"{e.id}: duplicate element id")
            seen.add(e.id)

            x, y, ew, eh = e.bbox
            if ew <= 0 or eh <= 0:
                problems.append(f"{e.id}: degenerate bbox {e.bbox}")
            if x < 0 or y < 0 or x + ew > w or y + eh > h:
                problems.append(f"{e.id}: bbox {e.bbox} lies outside the {w}x{h} display")
            if e.kind is ElementKind.NEEDLE and e.pivot is None:
                problems.append(f"{e.id}: needle element needs a pivot")
            if e.position.kind == "linear" and e.position.direction == (0.0, 0.0):
                problems.append(f"{e.id}: linear position model with zero travel")
            if e.position.kind != "static" and not e.template_path:
                problems.append(
                    f"{e.id}: moves with {e.signal or 'a signal'} but has no stored "
                    "template. The reference holds it at one state only, so at any "
                    "other state the expected box would be cropped from background "
                    "and the element reported as wrong content."
                )
            if e.search_margin_px <= e.tolerance.tol_fail:
                problems.append(
                    f"{e.id}: search margin {e.search_margin_px:.1f} px does not exceed "
                    f"tol_fail {e.tolerance.tol_fail:.1f} px, so a real failure would land "
                    "on the search border and be reported with an unknown magnitude"
                )
            complaint = e.tolerance.check_against_sigma()
            if complaint:
                problems.append(f"{e.id}: {complaint}")
            for other in e.occludes:
                if other not in {x.id for x in self.elements}:
                    problems.append(f"{e.id}: occludes unknown element {other!r}")
        return problems


def _safe_name(element_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in element_id)


def _spec_to_dict(e: ElementSpec, defaults: Tolerance) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": e.id,
        "kind": e.kind.value,
        "bbox": [round(float(v), 3) for v in e.bbox],
        "source": e.source,
    }
    if e.signal:
        d["signal"] = e.signal
    if e.template_path:
        d["template"] = e.template_path
    if e.template_anchor is not None:
        d["template_anchor"] = [float(v) for v in e.template_anchor]
    if e.position.kind != "static":
        d["position"] = asdict(e.position)
    if e.pivot is not None:
        d["pivot"] = [float(v) for v in e.pivot]
    if e.expected_angle_deg is not None:
        d["expected_angle_deg"] = float(e.expected_angle_deg)
    if e.angle is not None:
        d["angle"] = asdict(e.angle)
    if e.estimator != "auto":
        d["estimator"] = e.estimator
    if e.context_px is not None:
        d["context_px"] = int(e.context_px)
    if e.mask:
        d["mask"] = e.mask
    if e.occludes:
        d["occludes"] = list(e.occludes)
    if e.search_margin_px != ElementSpec.search_margin_px:
        d["search_margin_px"] = float(e.search_margin_px)
    # Only record what differs from the profile defaults, so a reviewer reading
    # the file sees the per-element decisions and not a wall of repetition.
    overrides = {
        k: v for k, v in asdict(e.tolerance).items() if v != asdict(defaults).get(k)
    }
    if overrides:
        d["tolerance"] = overrides
    if e.notes:
        d["notes"] = e.notes
    return d


def _spec_from_dict(d: Mapping[str, Any], defaults: Tolerance) -> ElementSpec:
    tol = Tolerance(**{**asdict(defaults), **(d.get("tolerance") or {})})
    if d.get("position"):
        raw = d["position"]
        pos = PositionModel(
            kind=raw.get("kind", "static"),
            origin=tuple(raw.get("origin", (0.0, 0.0))),
            direction=tuple(raw.get("direction", (0.0, 0.0))),
            value_min=raw.get("value_min", 0.0),
            value_max=raw.get("value_max", 1.0),
        )
    else:
        # ElementSpec.__post_init__ derives a static model from the bbox.
        pos = PositionModel()
    spec = ElementSpec(
        id=d["id"],
        kind=ElementKind(d.get("kind", "icon")),
        bbox=_tuple4(d["bbox"]),
        position=pos,
        tolerance=tol,
        source=d.get("source", "teachin"),
        signal=d.get("signal"),
        template_path=d.get("template"),
        template_anchor=tuple(d["template_anchor"]) if d.get("template_anchor") else None,
        pivot=tuple(d["pivot"]) if d.get("pivot") else None,
        expected_angle_deg=d.get("expected_angle_deg"),
        angle=AngleModel(**d["angle"]) if d.get("angle") else None,
        estimator=d.get("estimator", "auto"),
        context_px=d.get("context_px"),
        mask=d.get("mask"),
        occludes=list(d.get("occludes", [])),
        notes=d.get("notes", ""),
    )
    if "search_margin_px" in d:
        spec.search_margin_px = float(d["search_margin_px"])
    return spec


# --------------------------------------------------------------------------
# design-tool import
# --------------------------------------------------------------------------

DEFAULT_FIELD_MAP = {
    "id": "name",
    "x": "x",
    "y": "y",
    "width": "width",
    "height": "height",
    "children": "children",
    "kind": "type",
    "visible": "visible",
}

KIND_ALIASES = {
    "telltale": ElementKind.TELLTALE,
    "indicator": ElementKind.TELLTALE,
    "icon": ElementKind.ICON,
    "image": ElementKind.ICON,
    "symbol": ElementKind.ICON,
    "text": ElementKind.TEXT,
    "label": ElementKind.TEXT,
    "needle": ElementKind.NEEDLE,
    "pointer": ElementKind.NEEDLE,
    "gauge": ElementKind.NEEDLE,
}


def import_design_tree(
    node: Mapping[str, Any],
    *,
    screen: str,
    display_size: tuple[int, int],
    theme: str | None = None,
    field_map: Mapping[str, str] | None = None,
    defaults: Tolerance | None = None,
    include_invisible: bool = False,
) -> LayoutProfile:
    """Build a profile from an HMI design-tool node-tree export.

    The exports differ between tools, so the field names are configurable rather
    than guessed; ``field_map`` maps this function's vocabulary onto the export's
    keys.  Positions are accumulated down the tree because every tool this
    targets stores child positions relative to the parent.

    Only leaves with a non-zero extent become elements -- containers and layout
    groups are traversed, not measured.
    """
    fm = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    profile = LayoutProfile(
        screen=screen, display_size=display_size, theme=theme, defaults=defaults
    )

    def walk(n: Mapping[str, Any], ox: float, oy: float) -> None:
        x = ox + float(n.get(fm["x"], 0) or 0)
        y = oy + float(n.get(fm["y"], 0) or 0)
        w = float(n.get(fm["width"], 0) or 0)
        h = float(n.get(fm["height"], 0) or 0)
        children = n.get(fm["children"]) or []
        visible = n.get(fm["visible"], True)
        if (visible or include_invisible) and not children and w > 0 and h > 0:
            raw_kind = str(n.get(fm["kind"], "") or "").lower()
            kind = next(
                (v for k, v in KIND_ALIASES.items() if k in raw_kind), ElementKind.ICON
            )
            profile.add(
                ElementSpec(
                    id=str(n[fm["id"]]),
                    kind=kind,
                    bbox=(x, y, w, h),
                    position=PositionModel(origin=(x, y)),
                    tolerance=profile.defaults,
                    source="design",
                    notes=f"imported from design export ({raw_kind or 'untyped'})",
                )
            )
        for child in children:
            walk(child, x, y)

    walk(node, 0.0, 0.0)
    return profile

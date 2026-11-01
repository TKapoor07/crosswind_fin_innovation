"""Fin geometry candidates.

A-D reproduce the ``GEO`` table in the Three.js page (the proposal's candidate families).
``stall_rad`` is in RADIANS and ``a0_deg`` in DEGREES, exactly as in the page.
Only the toy model uses the aerodynamic fields (a0, k, cd0, stall); learned/tabulated models
use the dimensional descriptors (span, chord, thickness, camber).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

SPAN_RANGE = (0.2, 1.5)    # m  (the page's sliders use 0.5-1.0; wider here so CAD sizes can be tested)
CHORD_RANGE = (0.15, 1.0)  # m


@dataclass(frozen=True)
class FinGeometry:
    id: str
    label: str
    thickness: float          # t/c
    camber: float             # m/c
    a0_deg: float             # zero-lift angle (toy model)
    k: float                  # lift-slope factor (toy model)
    cd0: float                # parasitic drag (toy model)
    stall_rad: float          # stall angle (toy model)
    span_m: float
    chord_m: float
    mirror_camber: bool = True  # True: camber can be flipped to suit either wind side (what the page assumes)

    @property
    def area_m2(self) -> float:
        return self.span_m * self.chord_m

    @property
    def aspect_ratio(self) -> float:
        return self.span_m / self.chord_m

    def sized(self, span_m: float | None = None, chord_m: float | None = None) -> "FinGeometry":
        span = self.span_m if span_m is None else float(span_m)
        chord = self.chord_m if chord_m is None else float(chord_m)
        if not (SPAN_RANGE[0] <= span <= SPAN_RANGE[1]):
            raise ValueError(f"span_m {span} outside {SPAN_RANGE}")
        if not (CHORD_RANGE[0] <= chord <= CHORD_RANGE[1]):
            raise ValueError(f"chord_m {chord} outside {CHORD_RANGE}")
        return replace(self, span_m=span, chord_m=chord)

    def key(self) -> tuple:
        """Identity used to match learned/tabulated data to a request (size-sensitive)."""
        return (self.id, round(self.span_m, 4), round(self.chord_m, 4))


CANDIDATES: dict[str, FinGeometry] = {
    "A": FinGeometry("A", "A – Flat plate", .03, 0.0, 0.0, .85, .035, .15, .8, .5),
    "B": FinGeometry("B", "B – Symmetric airfoil", .12, 0.0, 0.0, 1.0, .014, .21, .8, .5),
    "C": FinGeometry("C", "C – Cambered airfoil", .12, .04, -4.0, 1.0, .018, .2, .8, .5),
    "D": FinGeometry("D", "D – High-aspect-ratio fin", .12, 0.0, 0.0, 1.0, .014, .21, 1.0, .35),
}


def resolve(geometry_id: str, span_m: float | None = None, chord_m: float | None = None) -> FinGeometry:
    try:
        base = CANDIDATES[geometry_id]
    except KeyError:
        raise ValueError(f"unknown fin geometry {geometry_id!r}; expected one of {sorted(CANDIDATES)}") from None
    return base.sized(span_m, chord_m)

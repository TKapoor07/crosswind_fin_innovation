"""Wind kinematics and sign conventions.

CONVENTIONS (identical to the Three.js page, written down once so nobody has to guess):

* Wind direction ``wind_from_deg`` is the direction the wind blows FROM, measured from the
  vehicle's nose, clockwise seen from above: 0 = headwind, 90 = from the right, 180 = tailwind,
  270 = from the left.
* Apparent wind is expressed as a *source* vector (where it comes from):
      sx = W cos(B) + v      (positive = air arriving from the front)
      sz = W sin(B)          (positive = air arriving from the right)
* ``gam_deg`` (0..180) is the angle between the nose and the apparent-wind source.
  ``side`` = +1 if the apparent wind comes from the right, -1 from the left.
* "Mirrored frame": everything aerodynamic is solved as if the wind came from the right
  (gam >= 0). The fin angle in that frame is ``d`` and AoA = gam - d. The physical fin
  angle (positive = leading edge turned toward the right) is ``side * d``.
  ``fy`` in the mirrored frame becomes a physical rightward force ``side * fy``.
* Vehicle axes for stored CFD data follow ISO 8855: +x forward, +y LEFT, +z up, yaw positive
  nose-left. (Different from the scene's +z = right, so the conversion lives in ``cfd``.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

D2R = math.pi / 180.0


@dataclass(frozen=True)
class Flow:
    """Apparent-wind state seen by the vehicle at one instant."""
    va: float            # apparent wind speed, m/s
    gam_deg: float       # 0..180, nose to apparent-wind source
    side: int            # +1 wind from right, -1 from left
    sx: float
    sz: float
    q: float             # dynamic pressure, Pa
    v_ms: float          # vehicle ground speed
    wind_ms: float       # true wind speed actually used (after gust)
    wind_from_deg: float # true wind direction actually used (after gust)

    @property
    def yaw_deg(self) -> float:
        """Signed apparent-wind angle (positive = from the right)."""
        return self.side * self.gam_deg


def apparent_wind(v_ms: float, wind_ms: float, wind_from_deg: float, *, side_hold: int = 1,
                  rho: float = 1.225, side_threshold: float = 0.05) -> Flow:
    b = wind_from_deg * D2R
    sx = wind_ms * math.cos(b) + v_ms
    sz = wind_ms * math.sin(b)
    va = math.hypot(sx, sz)
    gam = math.atan2(abs(sz), sx) / D2R
    side = (1 if sz > 0 else -1) if abs(sz) > side_threshold else side_hold
    q = 0.5 * rho * va * va
    return Flow(va, gam, side, sx, sz, q, v_ms, wind_ms, wind_from_deg)


def apply_gust(wind_ms: float, wind_from_deg: float, t: float) -> tuple[float, float]:
    """Deterministic gust used by the page's 'Gusty wind' checkbox (sum of two sinusoids)."""
    if wind_ms <= 0:
        return wind_ms, wind_from_deg
    w = max(0.0, wind_ms * (1 + 0.35 * (math.sin(t * 1.3) + 0.6 * math.sin(t * 3.1 + 1)) / 1.6))
    return w, wind_from_deg + 14 * math.sin(t * 0.9 + 2)


def to_mirrored(yaw_deg: float, fin_angle_deg: float) -> tuple[float, float, int]:
    """(signed yaw, physical fin angle) -> (gam, d, side)."""
    side = 1 if yaw_deg >= 0 else -1
    return abs(yaw_deg), side * fin_angle_deg, side


def apparent_to_wind(v_ms: float, va: float, yaw_deg: float) -> tuple[float, float]:
    """Inverse of ``apparent_wind``: which (true wind speed, wind-from angle) produce a given
    apparent wind (va, yaw) at ground speed v? Used to turn ML-proposed CFD cases into the
    (speed, crosswind, direction) triple the CFD team actually sets up."""
    sx = va * math.cos(yaw_deg * D2R)
    sz = va * math.sin(yaw_deg * D2R)
    wx, wz = sx - v_ms, sz
    return math.hypot(wx, wz), (math.atan2(wz, wx) / D2R) % 360.0


def relative_wind_deg(wind_from_compass_deg: float, heading_compass_deg: float) -> float:
    """Meteorological wind-from bearing + vehicle heading -> page convention (0 = headwind)."""
    return (wind_from_compass_deg - heading_compass_deg) % 360.0

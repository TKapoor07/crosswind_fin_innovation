"""Unvalidated low-order fin-only model. Use for UI/software integration only."""
from __future__ import annotations
import math
import numpy as np
from .base import AeroModel, AeroResult
from ..geometry import FinGeometry
from ..kinematics import D2R, Flow

class ToyModelV2(AeroModel):
    name = "toy_v2"
    provenance = "toy"
    validated = False
    notes = ("Low-order isolated-fin approximation; not CFD or measured HELIARK data.",
             "No whole-body fin interaction, Reynolds effects, or validated post-stall physics.")

    def coefficients(self, g: FinGeometry, alpha_deg: np.ndarray, side: int = 1):
        ar = max(g.aspect_ratio, 0.1)
        a0 = g.a0_deg if (side >= 0 or g.mirror_camber) else -g.a0_deg
        x = (alpha_deg-a0)*D2R
        ax = np.abs(x)
        slope = g.k*2*math.pi/(1+2/ar)
        post = ax > g.stall_rad
        # Conservative smooth fall-off after stall; still a toy approximation.
        cl_attached = slope*x
        cl = np.where(post, np.sign(x)*np.maximum(0.0, slope*g.stall_rad - 1.2*(ax-g.stall_rad)), cl_attached)
        cd = g.cd0 + cl*cl/(math.pi*ar*0.8) + np.where(post, 1.1*np.sin(ax)**2, 0.0)
        return cl, cd, post

    def predict(self, geom: FinGeometry, flow: Flow, d_deg: np.ndarray) -> AeroResult:
        d = np.asarray(d_deg, dtype=float)
        aoa = flow.gam_deg-d
        cl, cd, post = self.coefficients(geom, aoa, flow.side)
        qS = flow.q*geom.area_m2
        lift, drag = qS*cl, qS*cd
        r = flow.gam_deg*D2R
        fx = lift*math.sin(r)-drag*math.cos(r)
        fy = -lift*math.cos(r)-drag*math.sin(r)
        z = np.zeros_like(d)
        return AeroResult(d, aoa, fx, fy, z.copy(), z.copy(), z.copy(),
                          np.ones(d.shape, dtype=bool), post)

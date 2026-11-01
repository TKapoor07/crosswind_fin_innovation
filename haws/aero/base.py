"""Interface every aerodynamic model implements (toy, CFD table, ML surrogate).

All forces are INCREMENTS relative to the no-fin vehicle, in the mirrored frame
(see ``heliark.kinematics``): fx > 0 pushes the car forward, ``side * fy`` is the physical
rightward force. This matches the proposal's metric  dFx = Fx(with fin) - Fx(baseline).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..geometry import FinGeometry
from ..kinematics import Flow


@dataclass
class AeroResult:
    d: np.ndarray            # mirrored-frame fin angles [deg]
    aoa: np.ndarray          # gam - d [deg]
    fx: np.ndarray           # [N]
    fy: np.ndarray           # [N] mirrored frame
    fz: np.ndarray           # [N] vertical (up +)
    sigma_fx: np.ndarray     # 1-sigma predictive uncertainty [N] (0 for deterministic models)
    sigma_fy: np.ndarray
    in_domain: np.ndarray    # bool: is this point inside the data/validity range of the model?
    post_stall: np.ndarray   # bool: model flags the point as beyond its stall/validated-attached range
    mx_nm: np.ndarray | None = None  # ISO vehicle axes: roll moment about +x
    my_nm: np.ndarray | None = None  # pitch moment about +y
    mz_nm: np.ndarray | None = None  # yaw moment about +z

    def __len__(self) -> int:
        return int(self.d.shape[0])

    def finite(self) -> bool:
        arrays=[self.fx,self.fy,self.fz,self.sigma_fx,self.sigma_fy]
        arrays += [a for a in (self.mx_nm,self.my_nm,self.mz_nm) if a is not None]
        return all(np.isfinite(a).all() for a in arrays)


class AeroModel(ABC):
    name: str = "base"
    provenance: str = "unknown"      # 'toy' | 'cfd-table' | 'surrogate'
    validated: bool = False          # may the UI/controller call this an engineering result?
    notes: tuple[str, ...] = ()

    @abstractmethod
    def predict(self, geom: FinGeometry, flow: Flow, d_deg: np.ndarray) -> AeroResult:
        """Evaluate at the given mirrored-frame fin angles for one flow condition."""

    def info(self) -> dict:
        return {"name": self.name, "provenance": self.provenance, "validated": self.validated,
                "notes": list(self.notes)}

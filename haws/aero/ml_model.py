from __future__ import annotations
import numpy as np
import pandas as pd
from .base import AeroModel, AeroResult
from ..geometry import FinGeometry
from ..kinematics import Flow
from ..ml import Surrogate
from .cfd_table import FEATURES

class MLSurrogateModel(AeroModel):
    name="ml_surrogate"; provenance="surrogate"; validated=False
    def __init__(self, surrogate: Surrogate): self.surrogate=surrogate; self.validated=surrogate.validated
    def info(self): return self.surrogate.info()
    def predict(self, geom: FinGeometry, flow: Flow, d_deg: np.ndarray):
        rows=pd.DataFrame([{"geometry":geom.id,"span_m":geom.span_m,"chord_m":geom.chord_m,
                            "v_kmh":flow.v_ms*3.6,"wind_ms":flow.wind_ms,"wind_from_deg":flow.wind_from_deg,
                            "fin_angle_deg":float(d)} for d in d_deg],columns=FEATURES)
        mu,sd=self.surrogate.predict_rows(rows)
        # Stored data use +y LEFT; convert to mirrored +right convention.
        fy=-flow.side*mu[:,1]
        domain=self.surrogate.domain_mask(rows)
        return AeroResult(np.asarray(d_deg,float),flow.gam_deg-np.asarray(d_deg,float),mu[:,0],fy,mu[:,2],
                          sd[:,0],sd[:,1],domain,np.zeros(len(d_deg),dtype=bool),mu[:,3],mu[:,4],mu[:,5])

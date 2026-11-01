"""CFD lookup using local inverse-distance interpolation with strict domain checks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import AeroModel, AeroResult
from ..geometry import FinGeometry
from ..kinematics import Flow

FEATURES = ["geometry", "span_m", "chord_m", "v_kmh", "wind_ms", "wind_from_deg", "fin_angle_deg"]
TARGETS = ["fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm"]
REQUIRED = FEATURES + TARGETS

class CFDTableModel(AeroModel):
    name = "cfd_table"
    provenance = "cfd-table"
    validated = False
    notes = ("Interpolates only within the supplied CFD sample envelope.",
             "A dataset is not considered validated until convergence and provenance metadata are independently reviewed.")

    def __init__(self, frame: pd.DataFrame):
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing: raise ValueError(f"CFD CSV missing columns: {missing}")
        df = frame[REQUIRED].copy()
        for c in FEATURES[1:] + TARGETS: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if len(df) < 5: raise ValueError("At least five valid CFD rows are required")
        if not set(df.geometry.astype(str)).issubset(set("ABCDE")): raise ValueError("geometry must be A, B, C, D or E")
        self.frame = df.reset_index(drop=True)
        self.validated = False

    def predict(self, geom: FinGeometry, flow: Flow, d_deg: np.ndarray) -> AeroResult:
        out=[]; sig=[]; domains=[]
        v=flow.v_ms*3.6
        for angle in np.asarray(d_deg, dtype=float):
            sub=self.frame[self.frame.geometry.astype(str)==geom.id]
            if sub.empty:
                out.append([0.,0.,0.,0.,0.,0.]); sig.append([1e6,1e6,1e6,1e6,1e6,1e6]); domains.append(False); continue
            x=np.column_stack([sub.span_m,sub.chord_m,sub.v_kmh,sub.wind_ms,sub.wind_from_deg,sub.fin_angle_deg])
            q=np.array([geom.span_m,geom.chord_m,v,flow.wind_ms,flow.wind_from_deg,angle],float)
            # Scale by observed ranges to avoid units dominating distance.
            lo=x.min(axis=0); hi=x.max(axis=0); scale=np.maximum(hi-lo,1e-6)
            xn=(x-lo)/scale; qn=(q-lo)/scale
            inside=bool(np.all(q>=lo-1e-9) and np.all(q<=hi+1e-9))
            dist=np.linalg.norm(xn-qn,axis=1)
            k=min(8,len(dist)); ix=np.argpartition(dist,k-1)[:k]
            w=1/np.maximum(dist[ix],1e-5)**2; w=w/w.sum()
            vals=sub.iloc[ix][TARGETS].to_numpy(float)
            mu=np.sum(vals*w[:,None],axis=0)
            sd=np.sqrt(np.sum(w[:,None]*(vals-mu)**2,axis=0))
            out.append(mu); sig.append(sd); domains.append(inside)
        a=np.asarray(out); s=np.asarray(sig); d=np.asarray(d_deg,float)
        # Table forces stored in vehicle axes. Map lateral sign into mirrored convention.
        fy_m=-flow.side*a[:,1]
        return AeroResult(d, flow.gam_deg-d, a[:,0], fy_m, a[:,2], s[:,0], s[:,1],
                          np.asarray(domains), np.zeros(len(d),dtype=bool), a[:,3], a[:,4], a[:,5])

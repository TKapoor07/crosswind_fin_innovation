from __future__ import annotations
import io, os, threading
from pathlib import Path
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from .config import Settings, CONTROLLER, VEHICLE, EnergyAssumptions
from .session import ModelRegistry, SessionManager, StepInput
from .aero.toy import ToyModelV2
from .aero.cfd_table import CFDTableModel
from .aero.ml_model import MLSurrogateModel
from .ml import train_surrogate, load_surrogate
from .race import simulate_race

ROOT=Path(__file__).resolve().parent.parent
settings=Settings()
registry=ModelRegistry(); registry.register(ToyModelV2())
sessions=SessionManager(registry,settings.max_sessions,settings.session_ttl_s)
model_lock=threading.RLock()

class StepRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    v_kmh: float=Field(60,ge=0,le=120)
    wind_ms: float=Field(8,ge=0,le=60)
    wind_from_deg: float=Field(70,ge=-720,le=720)
    gust: bool=False
    geometry: Literal["A","B","C","D"]="B"
    span_m: float|None=Field(None,ge=.2,le=1.5)
    chord_m: float|None=Field(None,ge=.15,le=1.0)
    mode: Literal["auto","ret","man"]="auto"
    manual_deg: float=Field(15,ge=-30,le=30)
    side_force_limit_n: float=Field(200,gt=0,le=5000)
    fault: bool=False
    model: Literal["toy_v2","cfd_table","ml_surrogate"]="toy_v2"
    legacy: bool|None=None
    dt: float=Field(0,ge=0,le=10)

class ResetRequest(BaseModel): energy_only: bool=False

app=FastAPI(title="HELIARK Adaptive Wind Strategy API",version="1.0.0",
    description="Research/simulation backend. Not a safety-certified vehicle controller.")
origins=list(settings.cors_origins)
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=["GET","POST","DELETE"],allow_headers=["*"])

@app.get("/api/v1/health")
def health():
    return {"status":"ok","service":"HAWS","version":"1.0.0","active_sessions":len(sessions),"models":registry.names(),
            "engineering_status":"concept-stage; no HELIARK fin CFD/wind-tunnel validation supplied"}

@app.get("/api/v1/models")
def models(): return {"models":registry.info()}

@app.post("/api/v1/evaluate")
def evaluate(req: StepRequest):
    sid=None
    try:
        s=sessions.create(); sid=s.id
        with s.lock: return s.step(StepInput(**req.model_dump())).to_dict()
    except (ValueError,KeyError) as e: raise HTTPException(422,str(e))
    except Exception as e: raise HTTPException(500,f"simulation failed: {e}")
    finally:
        if sid: sessions.delete(sid)

@app.post("/api/v1/sessions")
def create_session():
    try:
        s=sessions.create(); return {"session_id":s.id,"expires_after_s":settings.session_ttl_s}
    except RuntimeError as e: raise HTTPException(429,str(e))

@app.post("/api/v1/sessions/{sid}/step")
def session_step(sid: str,req: StepRequest):
    try:
        s=sessions.get(sid)
        with s.lock: return s.step(StepInput(**req.model_dump())).to_dict()
    except KeyError as e: raise HTTPException(404,str(e))
    except (ValueError,TypeError) as e: raise HTTPException(422,str(e))

@app.post("/api/v1/sessions/{sid}/reset")
def reset_session(sid: str,req: ResetRequest):
    try:
        s=sessions.get(sid)
        with s.lock: s.reset(energy_only=req.energy_only)
        return {"reset":True,"energy_only":req.energy_only}
    except KeyError as e: raise HTTPException(404,str(e))

@app.delete("/api/v1/sessions/{sid}")
def delete_session(sid: str):
    if not sessions.delete(sid): raise HTTPException(404,"session not found")
    return {"deleted":True}

@app.post("/api/v1/data/cfd/upload")
async def upload_cfd(file: UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(415,"Upload a CSV file")
    raw=await file.read()
    if len(raw)>25*1024*1024: raise HTTPException(413,"CSV exceeds 25 MB")
    try:
        frame=pd.read_csv(io.BytesIO(raw)); model=CFDTableModel(frame)
        path=ROOT/"data"/"cfd"/"latest_cfd.csv"; path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)
        with model_lock: registry.register(model)
        return {"loaded":True,"rows":len(model.frame),"model":"cfd_table","validated":False,
                "columns":list(model.frame.columns),"note":"CSV loaded; verify CFD provenance, mesh convergence and held-out accuracy before engineering use."}
    except Exception as e: raise HTTPException(422,str(e))

@app.post("/api/v1/ml/train")
def train_ml():
    csv=ROOT/"data"/"cfd"/"latest_cfd.csv"
    if not csv.exists(): raise HTTPException(409,"Upload CFD data first at /api/v1/data/cfd/upload")
    try:
        metrics=train_surrogate(str(csv),str(ROOT/"models"/"haws_surrogate.joblib"))
        surrogate=load_surrogate(str(ROOT/"models"/"haws_surrogate.joblib"))
        with model_lock: registry.register(MLSurrogateModel(surrogate))
        return {"trained":True,"metrics":metrics,"validated":False,
                "note":"Training/test metrics are preliminary. Do not enable vehicle control from this model without independent validation."}
    except Exception as e: raise HTTPException(422,str(e))

@app.post("/api/v1/race/simulate")
def race(req: dict):
    try: return simulate_race(req)
    except Exception as e: raise HTTPException(422,str(e))

@app.websocket("/ws/sessions/{sid}")
async def websocket_session(ws: WebSocket,sid: str):
    await ws.accept()
    try: s=sessions.get(sid)
    except KeyError:
        await ws.send_json({"error":"unknown or expired session"}); await ws.close(code=1008); return
    try:
        while True:
            raw=await ws.receive_json()
            try:
                req=StepRequest.model_validate(raw)
                with s.lock: result=s.step(StepInput(**req.model_dump())).to_dict()
                await ws.send_json(result)
            except Exception as e: await ws.send_json({"error":str(e)})
    except WebSocketDisconnect: pass

@app.get("/")
def frontend():
    p=ROOT/"frontend"/"heliark_haws_v2.html"
    if p.exists(): return FileResponse(p)
    return {"message":"HAWS API running. Open /docs for API documentation."}

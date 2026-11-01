import numpy as np
from fastapi.testclient import TestClient
from haws.api import app
from haws.kinematics import apparent_wind
from haws.aero.toy import ToyModelV2
from haws.geometry import resolve

client=TestClient(app)

def test_apparent_wind_convention():
    f=apparent_wind(10,5,90)
    assert f.sz>0 and f.side==1 and 0<f.gam_deg<90
    f2=apparent_wind(10,5,270)
    assert f2.sz<0 and f2.side==-1

def test_toy_model_finite():
    f=apparent_wind(60/3.6,8,70)
    r=ToyModelV2().predict(resolve('B'),f,np.arange(-30,31,1))
    assert r.finite() and len(r)==61

def test_health_and_evaluate():
    assert client.get('/api/v1/health').status_code==200
    r=client.post('/api/v1/evaluate',json={"v_kmh":60,"wind_ms":8,"wind_from_deg":70,"geometry":"B","dt":0.1})
    assert r.status_code==200, r.text
    d=r.json(); assert d['model']['validated'] is False
    assert len(d['curve']['d'])==61

def test_session_and_fault_retracts():
    sid=client.post('/api/v1/sessions').json()['session_id']
    r=client.post(f'/api/v1/sessions/{sid}/step',json={"fault":True,"dt":0.1})
    assert r.status_code==200 and r.json()['state']=='SAFE / FAULT'
    assert r.json()['fin']['deployed'] is False
    assert client.delete(f'/api/v1/sessions/{sid}').status_code==200

def test_invalid_input_rejected():
    r=client.post('/api/v1/evaluate',json={"v_kmh":999})
    assert r.status_code==422


def test_cfd_table_and_ml_pipeline(tmp_path):
    import pandas as pd
    from haws.aero.cfd_table import CFDTableModel
    from haws.ml import train_surrogate, load_surrogate
    from haws.aero.ml_model import MLSurrogateModel
    rows=[]
    for g in "ABCD":
        for i in range(12):
            a=-30+i*5
            rows.append({"geometry":g,"span_m":.8,"chord_m":.5,"v_kmh":40+i*2,
                "wind_ms":5+i*.2,"wind_from_deg":60+i*3,"fin_angle_deg":a,
                "fx_n":5*np.sin(np.deg2rad(a))+i*.1,"fy_n":20*np.cos(np.deg2rad(a)),"fz_n":0,
                "mx_nm":1.2,"my_nm":2.3,"mz_nm":3.4,"case_group":f"{g}-{i//3}"})
    df=pd.DataFrame(rows)
    table=CFDTableModel(df)
    flow=apparent_wind(60/3.6,8,70)
    tr=table.predict(resolve("B"),flow,np.array([-10.,0.,10.]))
    assert tr.finite() and tr.mz_nm is not None
    csv=tmp_path/"cases.csv"; artifact=tmp_path/"surrogate.joblib"; df.to_csv(csv,index=False)
    metrics=train_surrogate(str(csv),str(artifact))
    model=MLSurrogateModel(load_surrogate(str(artifact)))
    pred=model.predict(resolve("B"),flow,np.array([-10.,0.,10.]))
    assert pred.finite() and pred.mx_nm is not None and len(pred)==3
    assert model.info()["validated"] is False
    assert metrics["n_total"]==48

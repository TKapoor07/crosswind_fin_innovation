"""Train/evaluate an interpretable tabular CFD surrogate; never silently promotes it to validated."""
from __future__ import annotations
import json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .aero.cfd_table import FEATURES, TARGETS

class Surrogate:
    name="ml_surrogate"; provenance="surrogate"
    def __init__(self, artifact: dict): self.artifact=artifact; self.pipeline=artifact["pipeline"]
    @property
    def validated(self): return bool(self.artifact.get("validated",False))
    def info(self):
        return {"name":self.name,"provenance":self.provenance,"validated":self.validated,
                "metrics":self.artifact.get("metrics"),"notes":["ExtraTrees CFD surrogate; valid only inside its training domain."]}
    def domain_mask(self, rows: pd.DataFrame):
        dom=self.artifact["domain"]
        ok=rows.geometry.astype(str).isin(dom["geometry"]).to_numpy()
        for col,lo in dom["numeric_min"].items():
            hi=dom["numeric_max"][col]
            ok &= (rows[col].to_numpy(float)>=lo-1e-9) & (rows[col].to_numpy(float)<=hi+1e-9)
        return ok
    def predict_rows(self, rows: pd.DataFrame):
        x=rows[FEATURES]
        trees=self.pipeline.named_steps["model"].estimators_
        # Multi-output ExtraTrees exposes one tree per estimator; estimate epistemic spread.
        preds=np.stack([t.predict(self.pipeline.named_steps["prep"].transform(x)) for t in trees],axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

def train_surrogate(csv_path: str, output_path: str, *, test_size=.2, random_state=42):
    df=pd.read_csv(csv_path)
    missing=[c for c in FEATURES+TARGETS if c not in df]
    if missing: raise ValueError(f"CSV missing required columns: {missing}")
    for c in FEATURES[1:]+TARGETS: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=FEATURES+TARGETS).copy()
    if len(df)<30: raise ValueError("At least 30 complete CFD rows are required for a preliminary train/test split")
    cats=["geometry"]; nums=FEATURES[1:]
    prep=ColumnTransformer([("geometry",OneHotEncoder(handle_unknown="ignore"),cats),("numeric","passthrough",nums)])
    pipe=Pipeline([("prep",prep),("model",ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=1.0,
                                                            random_state=random_state,n_jobs=-1))])
    if "case_group" in df.columns and df.case_group.nunique()>1:
        split=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
        tr,te=next(split.split(df[FEATURES],df[TARGETS],groups=df.case_group))
    else: tr,te=train_test_split(np.arange(len(df)),test_size=test_size,random_state=random_state)
    pipe.fit(df.iloc[tr][FEATURES],df.iloc[tr][TARGETS])
    pred=pipe.predict(df.iloc[te][FEATURES]); y=df.iloc[te][TARGETS].to_numpy()
    metrics={"n_total":int(len(df)),"n_train":int(len(tr)),"n_test":int(len(te)),
             "r2_per_target":{t:float(r2_score(y[:,i],pred[:,i])) for i,t in enumerate(TARGETS)},
             "rmse_per_target":{t:float(np.sqrt(mean_squared_error(y[:,i],pred[:,i]))) for i,t in enumerate(TARGETS)},
             "mae_per_target":{t:float(mean_absolute_error(y[:,i],pred[:,i])) for i,t in enumerate(TARGETS)}}
    # This is a quality report, not a safety certification. Needs independent CFD review and enough cases.
    artifact={"pipeline":pipe,"features":FEATURES,"targets":TARGETS,"metrics":metrics,
              "domain":{"geometry":sorted(df.geometry.astype(str).unique()),
                        "numeric_min":df[nums].min().to_dict(),"numeric_max":df[nums].max().to_dict()},
              "validated":False,"validation_note":"Requires independent holdout/mesh convergence review and approved engineering thresholds."}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(artifact,output_path)
    return metrics

def load_surrogate(path: str): return Surrogate(joblib.load(path))

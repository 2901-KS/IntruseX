import os, time, threading
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import joblib

from data.schema import runtime_columns
from models.models import MODEL_REGISTRY
from evaluation.attribution import get_provider, rule_hint
from intelligence.risk import risk_score, AdaptiveThresholdManager
from intelligence.narratives import make_narrative

MODEL_NAME=os.getenv("INTRUSEX_MODEL","tft")
CHECKPOINT=os.getenv("INTRUSEX_CHECKPOINT","artifacts/tft.pt")
DEVICE=torch.device(os.getenv("INTRUSEX_DEVICE","cpu"))

app=FastAPI(title="IntruSex-BH Inference API",version="1.0.0")
metrics_app=make_asgi_app()
app.mount("/metrics",metrics_app)

PREDICTIONS=Counter("intrusex_predictions_total","Number of inference requests")
ALERTS=Counter("intrusex_alerts_total","Number of generated alerts",["severity"])
LATENCY=Histogram("intrusex_inference_seconds","Inference latency in seconds")
RISK=Gauge("intrusex_risk_score","Latest risk score")
CONF=Gauge("intrusex_confidence","Latest model confidence")
STAGE=Gauge("intrusex_predicted_stage","Latest predicted stage")
NEXT_STAGE=Gauge("intrusex_predicted_next_stage","Latest predicted next stage")
SOJOURN=Gauge("intrusex_estimated_sojourn_polls","Estimated remaining polls")
STAGE_PROB=Gauge("intrusex_stage_probability","Current stage probability",["stage"])
NEXT_PROB=Gauge("intrusex_next_stage_probability","Next-stage probability",["stage"])
MODEL_INFO=Gauge("intrusex_model_info","Loaded model",["model","device"])
ACTIVE_SESSIONS=Gauge("intrusex_active_sessions","Current active session count")
SESSION_STATE=Gauge("intrusex_session_state","Latest session state",["session_id","stage","risk_level"])
TOP_EVIDENCE=Gauge("intrusex_top_evidence","Top evidence magnitude",["feature"])
ALERT_STATE=Gauge("intrusex_alert_state","Latest alert state",["session_id","stage","risk_level"])
LOCK=threading.Lock()

class PredictRequest(BaseModel):
    history: List[Dict[str,float]] = Field(..., min_length=1, description="Causal history, oldest to newest.")
    dpid: str = "network"
    session_id: Optional[str] = None
    trust_score: float = 0.0

class PredictResponse(BaseModel):
    model: str
    predicted_stage: int
    stage_probabilities: List[float]
    predicted_next_stage: int
    next_stage_probabilities: List[float]
    estimated_sojourn_remaining: float
    confidence: float
    risk_score: float
    risk_level: str
    alert_threshold: float
    alert: bool
    top_features: List[Dict[str,Any]]
    narrative: str

class Runtime:
    def __init__(self):
        if not os.path.exists(CHECKPOINT):
            raise RuntimeError(f"Checkpoint not found: {CHECKPOINT}")
        ckpt=torch.load(CHECKPOINT,map_location=DEVICE,weights_only=False)
        self.columns=list(ckpt["columns"])
        self.model_name=ckpt.get("model",MODEL_NAME)
        self.hidden=int(ckpt["hidden"])
        self.history=int(ckpt.get("history",32))
        self.scaler=joblib.load(os.getenv("INTRUSEX_SCALER","artifacts/scaler.joblib"))
        self.model=MODEL_REGISTRY[self.model_name](len(self.columns),self.hidden).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.provider=get_provider(self.model_name)
        self.thresholds=AdaptiveThresholdManager(base=100)
        MODEL_INFO.labels(self.model_name,str(DEVICE)).set(1)

    def _vectorize(self,history):
        missing=[]
        rows=[]
        for row in history:
            if not isinstance(row,dict): raise ValueError("Every history item must be an object.")
            missing += [c for c in self.columns if c not in row]
            rows.append([float(row[c]) for c in self.columns if c in row])
        if missing:
            missing=sorted(set(missing))
            raise ValueError(f"Missing required features: {missing}")
        df=pd.DataFrame(history,columns=self.columns)
        x=self.scaler.transform(df).astype("float32")
        if len(x)<self.history:
            pad=np.repeat(x[:1],self.history-len(x),axis=0)
            x=np.concatenate([pad,x],axis=0)
        else:
            x=x[-self.history:]
        return torch.from_numpy(x[None,...]).to(DEVICE), df

    def predict(self,req):
        x,raw=self._vectorize(req.history)
        start=time.perf_counter()
        with torch.no_grad():
            out=self.model(x)
            sp=torch.softmax(out["stage_logits"],-1)[0]
            np_=torch.softmax(out["next_stage_logits"],-1)[0]
            stage=int(sp.argmax())
            nxt=int(np_.argmax())
            soj=float(torch.nn.functional.softplus(out["sojourn_params"][0,0]).cpu())
            conf=float(sp.max())
        elapsed=time.perf_counter()-start
        LATENCY.observe(elapsed); PREDICTIONS.inc()

        # Contextual change signals are computed only from supplied telemetry.
        ent_drop=0.0
        qdr_drop=0.0
        if "entropy" in raw:
            v=raw["entropy"].to_numpy(float)
            if len(v)>1:
                base=np.median(v[:-1]); scale=np.median(np.abs(v[:-1]-base))+1e-6
                ent_drop=float(np.clip((base-v[-1])/(3*scale),0,1))
        qcols=[c for c in raw.columns if c.startswith("qdrm_") or c.startswith("qdrv_")]
        if qcols:
            v=raw[qcols].mean(axis=1).to_numpy(float)
            if len(v)>1:
                base=np.median(v[:-1]); scale=np.median(np.abs(v[:-1]-base))+1e-6
                qdr_drop=float(np.clip((base-v[-1])/(3*scale),0,1))
        risk,level=risk_score(stage,ent_drop,qdr_drop,nxt,conf)
        threshold=self.thresholds.threshold(req.trust_score,stage,conf)
        alert=bool(risk>threshold)
        if alert: ALERTS.labels(level).inc()
        sid=req.session_id or req.dpid
        SESSION_STATE.labels(sid,str(stage),level).set(1)
        ALERT_STATE.labels(sid,str(stage),level).set(1 if alert else 0)
        RISK.set(risk); CONF.set(conf); STAGE.set(stage); NEXT_STAGE.set(nxt); SOJOURN.set(soj)
        for i,p in enumerate(sp.cpu().numpy()): STAGE_PROB.labels(f"S{i}").set(float(p))
        for i,p in enumerate(np_.cpu().numpy()): NEXT_PROB.labels(f"S{i}").set(float(p))

        # Attribution is performed outside the no-grad block.
        top=[]
        try:
            attr=self.provider.explain(self.model,x)
            if "feature_time_abs_grad" in attr:
                vals=attr["feature_time_abs_grad"][0].mean(0).numpy()
            elif "feature_time_importance" in attr:
                vals=attr["feature_time_importance"][0].mean(0).numpy()
            else: vals=np.zeros(len(self.columns))
            order=np.argsort(vals)[::-1][:8]
            top=[{"feature":self.columns[int(i)],"magnitude":float(vals[int(i)])} for i in order]
        except Exception:
            top=rule_hint(self.columns,x[0,-1])
        for item in top[:8]:
            TOP_EVIDENCE.labels(item["feature"]).set(item["magnitude"])
        narrative=make_narrative(stage,nxt,conf,risk,level,top)
        return PredictResponse(
            model=self.model_name,predicted_stage=stage,
            stage_probabilities=sp.cpu().tolist(),predicted_next_stage=nxt,
            next_stage_probabilities=np_.cpu().tolist(),
            estimated_sojourn_remaining=soj,confidence=conf,
            risk_score=risk,risk_level=level,alert_threshold=threshold,
            alert=alert,top_features=top,narrative=narrative
        )

runtime: Runtime|None=None

@app.on_event("startup")
def startup():
    global runtime
    runtime=Runtime()

@app.get("/health")
def health():
    return {"status":"ok","model":MODEL_NAME,"device":str(DEVICE)}

@app.get("/ready")
def ready():
    if runtime is None: raise HTTPException(503,"model not loaded")
    return {"ready":True,"model":runtime.model_name,"features":len(runtime.columns)}

@app.get("/model")
def model_info():
    if runtime is None: raise HTTPException(503,"model not loaded")
    return {"model":runtime.model_name,"history":runtime.history,"feature_count":len(runtime.columns),
            "features":runtime.columns,"device":str(DEVICE)}

@app.post("/predict",response_model=PredictResponse)
def predict(req:PredictRequest):
    if runtime is None: raise HTTPException(503,"model not loaded")
    try: return runtime.predict(req)
    except ValueError as e: raise HTTPException(422,str(e))

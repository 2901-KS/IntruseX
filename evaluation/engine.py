import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from .metrics import stage_metrics, ordered_stage_score, sojourn_score, bootstrap_run_ci
from .calibration import TemperatureScaler, calibration_metrics
from .robustness import robustness_report
from intelligence.risk import risk_score
from .attribution import explain

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval(); rows=[]
    for b in loader:
        x=b["x"].to(device); out=model(x)
        sp=torch.softmax(out["stage_logits"],-1).cpu().numpy()
        np_=torch.softmax(out["next_stage_logits"],-1).cpu().numpy()
        pred=sp.argmax(1)
        next_pred=np_.argmax(1)
        dur=torch.nn.functional.softplus(out["sojourn_params"][:,0]).cpu().numpy()
        for i in range(len(pred)):
            rows.append({"run_id":str(b["run_id"][i]),"stage":int(b["stage"][i]),"pred":int(pred[i]),
                         "proba":sp[i],"next_stage":int(b["next_stage"][i]),"next_pred":int(next_pred[i]),
                         "next_proba":np_[i],"sojourn":float(b["sojourn"][i]),"pred_sojourn":float(dur[i]),
                         "confidence":float(sp[i].max())})
    return rows

def evaluate_model(model, loaders, device, model_name="model"):
    start=time.perf_counter(); rows=collect_predictions(model,loaders["test"],device); elapsed=time.perf_counter()-start
    y=np.array([r["stage"] for r in rows]); p=np.array([r["pred"] for r in rows])
    m=stage_metrics(y,p); m["ordered_stage_score"]=ordered_stage_score(y,p)
    nm=[r for r in rows if r["next_stage"]>=0]
    if nm: m["next_macro_f1"]=float(f1_score([r["next_stage"] for r in nm],[r["next_pred"] for r in nm],average="macro",zero_division=0))
    m["sojourn_score"]=sojourn_score([r["sojourn"] for r in nm],[r["pred_sojourn"] for r in nm]) if nm else None
    # Calibration is fitted on validation, then frozen before test.
    vr=collect_predictions(model,loaders["val"],device)
    cal=None
    if vr:
        # Fit on validation logits reconstructed from probabilities, then freeze before test.
        logits=np.log(np.stack([r["proba"] for r in vr]).clip(1e-6,1))
        cal=TemperatureScaler().fit(logits,[r["stage"] for r in vr])
        tp=cal.predict_proba(np.log(np.stack([r["proba"] for r in rows]).clip(1e-6,1)))
        m.update(calibration_metrics(y,tp)); m["temperature"]=float(cal.temperature.detach())
    m["latency_ms_per_window"]=1000*elapsed/max(1,len(rows)); m["latency_score"]=float(np.clip(1/(1+m["latency_ms_per_window"]),0,1))
    # Run-level CIs: aggregate each run before bootstrap.
    byrun={}
    for r in rows: byrun.setdefault(r["run_id"],[[],[]]); byrun[r["run_id"]][0].append(r["stage"]); byrun[r["run_id"]][1].append(r["pred"])
    run_f1=[stage_metrics(v[0],v[1])["macro_f1"] for v in byrun.values()]
    m["run_macro_f1_ci"]=bootstrap_run_ci(run_f1)
    # Operational utility using the native risk ladder.
    risk=[]
    for r in rows:
        score,_=risk_score(r["pred"],next_stage=r["next_pred"],confidence=r["confidence"]); risk.append(score)
    m["alert_utility"]=float(np.clip(np.mean((np.asarray(risk)>=60)==(y>=1)),0,1))
    return m, rows

def attach_robustness(model, loaders, device, metrics):
    rr=robustness_report(model,loaders["test"],device)
    metrics["robustness"]=rr; metrics["real_robustness"]=None  # must remain None until real Mininet/Ryu data exist
    metrics["synthetic_shift_robustness"]=rr["worst_macro_f1"]
    return metrics

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score, confusion_matrix

def stage_metrics(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2,3],zero_division=0)
    return {
        "accuracy":float(np.mean(y_true==y_pred)),
        "macro_f1":float(np.mean(f)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "mcc":float(matthews_corrcoef(y_true,y_pred)),
        "per_stage":{f"S{i}":{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i])} for i in range(4)},
        "confusion_matrix":confusion_matrix(y_true,y_pred,labels=[0,1,2,3]).tolist(),
        "s1_recall":float(r[1]),"s2_recall":float(r[2]),
    }

def ordered_stage_score(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    return float(1.0-np.mean(np.abs(y_true-y_pred))/3.0)

def sojourn_score(y_true,y_pred):
    if len(y_true)==0:return None
    mae=float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))
    return float(1.0/(1.0+mae))

def alert_utility_score(y_true, risk_scores, threshold=60):
    y=np.asarray(y_true); r=np.asarray(risk_scores)
    alert=r>=threshold
    attack=y>=1
    recall=float((alert & attack).sum()/max(1,attack.sum()))
    false=float((alert & ~attack).sum()/max(1,(~attack).sum()))
    return float(np.clip(recall-0.5*false,0,1))

def ranking_score(m):
    # Exact 100-point weights. Missing components are reported as missing, not
    # silently converted into fake evidence; score is normalized over populated
    # components while completeness is exposed separately.
    weights={"macro_f1":20,"s1_recall":5,"s2_recall":5,"ordered_stage_score":5,
             "next_macro_f1":20,"sojourn_score":10,"calibration_score":10,
             "alert_utility":10,"real_robustness":10,"latency_score":5}
    vals=[]; populated=0
    for k,w in weights.items():
        v=m.get(k)
        if v is not None:
            vals.append(float(v)*w); populated+=w
    return None if not vals else float(sum(vals)/populated*100)

def ranking_completeness(m):
    weights={"macro_f1":20,"s1_recall":5,"s2_recall":5,"ordered_stage_score":5,"next_macro_f1":20,
             "sojourn_score":10,"calibration_score":10,"alert_utility":10,"real_robustness":10,"latency_score":5}
    return float(sum(w for k,w in weights.items() if m.get(k) is not None))

def bootstrap_run_ci(run_values, n_boot=1000, seed=42):
    vals=np.asarray(run_values,dtype=float)
    if len(vals)<2:return {"mean":float(np.mean(vals)) if len(vals) else None,"low":None,"high":None,"n_runs":int(len(vals))}
    rng=np.random.default_rng(seed); means=[]
    for _ in range(n_boot): means.append(float(np.mean(rng.choice(vals,size=len(vals),replace=True))))
    return {"mean":float(np.mean(vals)),"low":float(np.percentile(means,2.5)),"high":float(np.percentile(means,97.5)),"n_runs":int(len(vals))}

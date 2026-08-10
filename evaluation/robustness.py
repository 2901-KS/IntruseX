import numpy as np
import torch
from evaluation.metrics import stage_metrics

def perturb(x, kind, seed=42):
    rng=np.random.default_rng(seed); z=x.copy()
    if kind=="traffic_scale": z*=1.0+rng.uniform(-0.25,0.35,size=(1,1,z.shape[-1]))
    elif kind=="timing_noise": z+=rng.normal(0,0.08,z.shape)
    elif kind=="missing_poll":
        if z.shape[1]>4:
            mask=rng.random(z.shape[:2])<0.08; z[mask]=0
    elif kind=="selective_drop_proxy":
        z[:,:,-min(3,z.shape[-1]):]*=0.25
    elif kind=="flash_crowd":
        z[:,:,0:min(3,z.shape[-1])]*=1.8
    return z

def robustness_report(model, loader, device, scenarios=None):
    scenarios=scenarios or ["traffic_scale","timing_noise","missing_poll","selective_drop_proxy","flash_crowd"]
    batches=list(loader); report={}
    for kind in scenarios:
        ys=[]; ps=[]
        for b in batches:
            x=b["x"].numpy(); xp=torch.tensor(perturb(x,kind),dtype=torch.float32,device=device)
            with torch.no_grad(): p=model(xp)["stage_logits"].argmax(-1).cpu().numpy()
            ys.extend(b["stage"].numpy()); ps.extend(p)
        report[kind]=stage_metrics(np.array(ys),np.array(ps))
    report["worst_macro_f1"]=float(min(v["macro_f1"] for v in report.values()))
    return report

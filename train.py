import argparse, json, os, random
import numpy as np, pandas as pd, torch, joblib
from torch.utils.data import DataLoader
from data.dataset import make_datasets
from data.schema import runtime_columns
from models.models import MODEL_REGISTRY
from heads.multitask import common_loss
from evaluation.engine import evaluate_model, attach_robustness

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--model",default="tft",choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--history",type=int,default=32)
    ap.add_argument("--stride",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="artifacts")
    a=ap.parse_args(); seed_all(a.seed)
    df=pd.read_csv(a.data)
    ds,scaler,split=make_datasets(df,a.history,a.stride,a.seed)
    loaders={k:DataLoader(v,batch_size=a.batch_size,shuffle=(k=="train"),num_workers=0) for k,v in ds.items()}
    d=len(runtime_columns(df))
    model=MODEL_REGISTRY[a.model](d,a.hidden)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    y=np.array([int(ds["train"][i]["stage"]) for i in range(len(ds["train"]))])
    counts=np.bincount(y,minlength=4)
    cw=torch.tensor(np.clip(len(y)/(4*np.maximum(counts,1)),0.5,8),dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=-1; best_state=None; epochs=[]
    for ep in range(1,a.epochs+1):
        model.train(); losses=[]
        for b in loaders["train"]:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
            out=model(batch["x"],stage_target=batch["stage"])
            loss,parts=common_loss(out,batch,feature_names=runtime_columns(df))
            unweighted=torch.nn.functional.cross_entropy(out["stage_logits"],batch["stage"])
            weighted=torch.nn.functional.cross_entropy(out["stage_logits"],batch["stage"],weight=cw)
            loss=loss-unweighted+weighted
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
            losses.append(float(loss.detach()))
        val,_=evaluate_model(model,{"val":loaders["val"],"test":loaders["test"]},device,a.model)
        rec={"epoch":ep,"train_loss":float(np.mean(losses)),"val_macro_f1":val["macro_f1"]}
        epochs.append(rec); print(rec)
        if rec["val_macro_f1"]>best:
            best=rec["val_macro_f1"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test,_=evaluate_model(model,loaders,device,a.model)
    test=attach_robustness(model,loaders,device,test)
    os.makedirs(a.out,exist_ok=True)
    torch.save({"model":a.model,"hidden":a.hidden,"history":a.history,"columns":runtime_columns(df),
                "split":split.__dict__,"seed":a.seed,"model_state":best_state},os.path.join(a.out,f"{a.model}.pt"))
    joblib.dump(scaler,os.path.join(a.out,"scaler.joblib"))
    with open(os.path.join(a.out,f"{a.model}.json"),"w") as f:
        json.dump({"model":a.model,"seed":a.seed,"epochs":epochs,"test":test,"split":split.__dict__},f,indent=2)
    print(json.dumps(test,indent=2))

if __name__=="__main__": main()

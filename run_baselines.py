import argparse, json, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data.dataset import make_datasets
from data.schema import runtime_columns
from baselines.models import RandomForestBaseline, FTTransformerBaseline, GRUBaseline
from baselines.semimarkov import FeatureWeightedSemiMarkov
from evaluation.metrics import stage_metrics, ordered_stage_score, sojourn_score

def arrays(ds):
    X=[]; y=[]; nxt=[]; dur=[]
    for i in range(len(ds)):
        b=ds[i]; X.append(b["x"].numpy()); y.append(int(b["stage"])); nxt.append(int(b["next_stage"])); dur.append(float(b["sojourn"]))
    return np.stack(X),np.array(y),np.array(nxt),np.array(dur)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--history",type=int,default=32); ap.add_argument("--stride",type=int,default=4); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--out",default="runs/baselines")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    df=pd.read_csv(a.data); ds,scaler,split=make_datasets(df,a.history,a.stride,42)
    Xtr,ytr,ntr,dtr=arrays(ds["train"]); Xv,yv,nv,dv=arrays(ds["val"]); Xt,yt,nt,dt=arrays(ds["test"])
    result={}
    rf=RandomForestBaseline().fit(Xtr[:,-1],ytr); yp=rf.predict(Xt[:,-1]); result["random_forest"]={**stage_metrics(yt,yp),"ordered_stage_score":ordered_stage_score(yt,yp)}
    sm=FeatureWeightedSemiMarkov().fit(Xtr[:,-1],ytr,ntr,dtr,runtime_columns(df)); np_,pp,dd=sm.predict(Xt[:,-1],yt); result["semi_markov"]={**stage_metrics(nt[nt>=0],np_[nt>=0]),"sojourn_score":sojourn_score(dt[nt>=0],dd[nt>=0])}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name,Cls in [("ft_transformer",FTTransformerBaseline),("gru",GRUBaseline)]:
        model=Cls(Xtr.shape[-1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3)
        loader=DataLoader([(torch.tensor(x),torch.tensor(y)) for x,y in zip(Xtr,ytr)],batch_size=64,shuffle=True)
        for _ in range(a.epochs):
            model.train()
            for xb,yb in loader:
                xb=xb.to(device); yb=yb.to(device); logits=model(xb); loss=torch.nn.functional.cross_entropy(logits,yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad(): pred=model(torch.tensor(Xt,dtype=torch.float32,device=device)).argmax(-1).cpu().numpy()
        result[name]={**stage_metrics(yt,pred),"ordered_stage_score":ordered_stage_score(yt,pred)}
    json.dump(result,open(os.path.join(a.out,"baseline_leaderboard.json"),"w"),indent=2); print(json.dumps(result,indent=2))
if __name__=="__main__": main()

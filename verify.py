import os, subprocess, sys, pandas as pd, torch
from data.schema import runtime_columns
from models.models import MODEL_REGISTRY
CSV="data/raw/intrusex_bh.csv"

def main():
    df=pd.read_csv(CSV)
    print("rows:",len(df),"runs:",df.run_id.nunique(),"runtime_features:",len(runtime_columns(df)))
    print("stage_counts:",df.stage.value_counts().sort_index().to_dict())
    x=torch.randn(2,8,len(runtime_columns(df)))
    for name,Cls in MODEL_REGISTRY.items():
        m=Cls(x.size(-1),24)
        with torch.no_grad(): o=m(x)
        print(f"{name:16s} OK stage={tuple(o['stage_logits'].shape)} next={tuple(o['next_stage_logits'].shape)}")
    print("verification complete")

if __name__=="__main__": main()

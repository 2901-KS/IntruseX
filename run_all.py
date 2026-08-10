import argparse, subprocess, sys, os
from models.models import MODEL_REGISTRY
from evaluation.leaderboard import build_leaderboard

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--history",type=int,default=32)
    ap.add_argument("--stride",type=int,default=8)
    ap.add_argument("--out",default="runs")
    ap.add_argument("--models",default="")
    a=ap.parse_args()
    names=sorted(MODEL_REGISTRY) if not a.models else [x.strip() for x in a.models.split(",") if x.strip()]
    os.makedirs(a.out,exist_ok=True)
    for seed in (42,43):
        for name in names:
            print(f"\n{'='*70}\n{name} seed={seed}\n{'='*70}")
            cmd=[sys.executable,"train.py","--data",a.data,"--model",name,
                 "--epochs",str(a.epochs),"--seed",str(seed),"--history",str(a.history),
                 "--stride",str(a.stride),"--out",a.out]
            subprocess.run(cmd,check=True)
    rows=build_leaderboard(a.out)
    print(f"leaderboard rows: {len(rows)}")

if __name__=="__main__": main()

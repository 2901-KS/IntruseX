import argparse, subprocess, sys, os
p=argparse.ArgumentParser()
p.add_argument("--model",default="tft")
p.add_argument("--epochs",type=int,default=5)
p.add_argument("--history",type=int,default=32)
p.add_argument("--stride",type=int,default=8)
a=p.parse_args()
cmd=[sys.executable,"train.py","--data","data/raw/intrusex_bh.csv","--model",a.model,
     "--epochs",str(a.epochs),"--history",str(a.history),"--stride",str(a.stride),
     "--out","artifacts"]
subprocess.run(cmd,check=True)
print("Deployment artifact ready:",os.path.abspath(f"artifacts/{a.model}.pt"))

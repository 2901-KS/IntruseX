import json, urllib.request, pandas as pd

API="http://127.0.0.1:8000/predict"
df=pd.read_csv("data/raw/intrusex_bh.csv")
features=[
"rate_H","advr_H","imb_H","grad_H","corr_H","dacc_H","qdrm_H","qdrv_H","skew_H","kurt_H",
"rate_G","advr_G","imb_G","grad_G","corr_G","dacc_G","qdrm_G","qdrv_G","skew_G","kurt_G",
"rate_I","advr_I","imb_I","grad_I","corr_I","dacc_I","qdrm_I","qdrv_I","skew_I","kurt_I",
"adv_rate","entropy","jsd"
]
history=df[features].head(32).to_dict(orient="records")
payload=json.dumps({"dpid":"network-demo","history":history}).encode()
req=urllib.request.Request(API,data=payload,headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=30) as r:
    print(r.read().decode())

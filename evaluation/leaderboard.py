import json, os
from evaluation.metrics import ranking_score, ranking_completeness

def build_leaderboard(run_dir="runs"):
    rows=[]
    for fn in sorted(os.listdir(run_dir)):
        if not fn.endswith(".json") or fn=="leaderboard.json": continue
        data=json.load(open(os.path.join(run_dir,fn)))
        if "test" not in data: continue
        m=data["test"]; m["model"]=data.get("model",fn[:-5]); m["ranking_score"]=ranking_score(m); m["ranking_completeness"]=ranking_completeness(m)
        rows.append(m)
    rows.sort(key=lambda x: (-1 if x["ranking_score"] is None else -x["ranking_score"]))
    out=os.path.join(run_dir,"leaderboard.json"); json.dump(rows,open(out,"w"),indent=2); return rows

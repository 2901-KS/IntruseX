import numpy as np
import pandas as pd

VALID_FORWARD={(0,1),(1,2),(2,3)}
def build_sessions(records, feature_cols, max_gap=2, similarity_threshold=3.0):
    """Build persistent behavioural sessions from classified telemetry."""
    df=records.sort_values(["dpid","timestamp"]).copy()
    sessions=[]; current=None; sid=0
    for _,row in df.iterrows():
        dpid=str(row.get("dpid","network"))
        stage=int(row["stage"])
        x=np.asarray([float(row[c]) for c in feature_cols],dtype=float)
        if current is None or current["dpid"]!=dpid:
            if current: sessions.append(current)
            sid+=1
            current={"session_id":f"S{sid}","dpid":dpid,"rows":[],"anomalous_recovery_jump":False}
        rows=current["rows"]
        new=False
        if rows:
            prev=rows[-1]
            gap=int(row.get("poll_index",0))-int(prev.get("poll_index",0)) if "poll_index" in row else 1
            prev_stage=int(prev["stage"])
            valid=(stage==prev_stage or (prev_stage,stage) in VALID_FORWARD)
            dist=float(np.linalg.norm(x-prev["_x"]))
            new=gap>max_gap or dist>similarity_threshold or not valid
            if prev_stage==3 and stage==0 and not bool(row.get("deactivate_event",False)):
                current["anomalous_recovery_jump"]=True
        if new:
            sessions.append(current); sid+=1
            current={"session_id":f"S{sid}","dpid":dpid,"rows":[],"anomalous_recovery_jump":False}
        rec=row.to_dict(); rec["_x"]=x; current["rows"].append(rec)
    if current: sessions.append(current)
    out=[]
    for s in sessions:
        rows=s["rows"]
        out.append({
            "session_id":s["session_id"],
            "dpid":s["dpid"],
            "stage_sequence":[int(r["stage"]) for r in rows],
            "feature_history":[r["_x"].tolist() for r in rows],
            "single_stage":len({int(r["stage"]) for r in rows})==1,
            "anomalous_recovery_jump":s["anomalous_recovery_jump"],
        })
    return out

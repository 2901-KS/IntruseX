import pandas as pd
from intelligence.sessions import build_sessions
from intelligence.risk import risk_score, alert_threshold

def test_session_and_risk():
    df=pd.DataFrame({"dpid":["s7"]*4,"timestamp":[1,2,3,4],"poll_index":[0,1,2,3],"stage":[0,1,2,3],"f":[0.,1.,2.,3.]})
    sessions=build_sessions(df,["f"])
    assert len(sessions)==1
    assert sessions[0]["stage_sequence"]==[0,1,2,3]
    score,level=risk_score(2,next_stage=3,confidence=.9)
    assert score>0 and level in {"HIGH","CRITICAL"}
    assert 20<=alert_threshold(0)<=200

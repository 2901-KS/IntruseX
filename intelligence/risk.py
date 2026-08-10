import numpy as np

STAGE_SEVERITY=np.array([0.0,25.0,60.0,100.0])

def risk_score(stage, entropy_drop=0.0, qdr_drop=0.0, next_stage=None, confidence=1.0):
    stage=int(stage)
    base=float(STAGE_SEVERITY[stage])
    context=15*np.clip(entropy_drop,0,1)+15*np.clip(qdr_drop,0,1)
    boost=20.0 if next_stage in (2,3) else 0.0
    score=(base+context+boost)*float(np.clip(confidence,0,1))
    level="LOW" if score<30 else "MEDIUM" if score<60 else "HIGH" if score<80 else "CRITICAL"
    return float(score),level

def alert_threshold(trust_score, stage_effect=0.0, confidence_effect=0.0, base=100.0):
    return float(np.clip(base+trust_score-stage_effect-confidence_effect,20,200))

class AdaptiveThresholdManager:
    def __init__(self,base=100.0): self.base=base
    def threshold(self,trust_score,stage,confidence):
        return alert_threshold(trust_score,10*stage,20*confidence,self.base)
    def decide(self,risk,trust_score,stage,confidence):
        t=self.threshold(trust_score,stage,confidence)
        return bool(risk>t),t

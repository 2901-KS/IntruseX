from .risk import alert_threshold

class AdaptiveThresholdManager:
    def __init__(self,base=100.0): self.base=base
    def threshold(self,trust_score,stage,confidence):
        return alert_threshold(trust_score,10*stage,20*confidence,self.base)
    def decide(self,risk_score_value,trust_score,stage,confidence):
        t=self.threshold(trust_score,stage,confidence)
        return bool(risk_score_value>t),t

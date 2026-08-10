import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import weibull_min

class FeatureWeightedSemiMarkov:
    """Mandatory Box-D baseline using feature-weighted transitions and fitted dwell distributions."""
    def __init__(self):
        self.transition_models={}; self.duration={}; self.feature_names=[]
    def fit(self,X,stage,next_stage,sojourn,feature_names=None):
        self.feature_names=list(feature_names or [])
        X=np.asarray(X); stage=np.asarray(stage); next_stage=np.asarray(next_stage); sojourn=np.asarray(sojourn)
        for s in range(3):
            m=(stage==s)&(next_stage>=0)
            if m.sum()<1: continue
            classes=np.unique(next_stage[m]).astype(int)
            if len(classes)>1:
                self.transition_models[s]=LogisticRegression(max_iter=1000).fit(X[m],next_stage[m])
            else:
                self.transition_models[s]=int(classes[0])
        for s in range(4):
            d=sojourn[(stage==s)&(sojourn>0)]
            if len(d):
                d=np.asarray(d,float)
                try:
                    shape,loc,scale=weibull_min.fit(d,floc=0)
                    self.duration[s]={"type":"weibull","shape":float(shape),"scale":float(scale),
                                      "mean":float(weibull_min.mean(shape,loc=loc,scale=scale))}
                except Exception:
                    self.duration[s]={"type":"empirical","mean":float(d.mean()),"median":float(np.median(d))}
        return self
    def predict_next_proba(self,X,stage):
        X=np.asarray(X); stage=np.asarray(stage); out=np.zeros((len(X),4))
        for s in range(4):
            idx=np.flatnonzero(stage==s); model=self.transition_models.get(s)
            if len(idx)==0: continue
            if hasattr(model,"predict_proba"):
                p=model.predict_proba(X[idx])
                for col,c in enumerate(model.classes_.astype(int)): out[idx,c]=p[:,col]
            elif isinstance(model,(int,np.integer)):
                out[idx,int(model)]=1.0
            else: out[idx,s]=1.0
        return out
    def predict(self,X,stage):
        p=self.predict_next_proba(X,stage)
        nxt=p.argmax(1)
        dur=np.array([self.duration.get(int(s),{}).get("mean",0.0) for s in stage])
        return nxt,p,dur

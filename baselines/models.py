import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier

class RandomForestBaseline:
    def __init__(self, seed=42, n_estimators=300):
        self.model=RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=seed, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return self.model.predict(X)
    @property
    def feature_importances_(self): return self.model.feature_importances_

class FTTransformerBaseline(nn.Module):
    """Compact FT-Transformer-style tabular baseline for the current-stage task."""
    def __init__(self, d, h=96, heads=4, layers=2):
        super().__init__()
        self.proj=nn.Linear(d,h)
        enc=nn.TransformerEncoderLayer(h,heads,batch_first=True,dropout=0.1)
        self.encoder=nn.TransformerEncoder(enc,layers)
        self.cls=nn.Linear(h,4)
    def forward(self,x):
        # x: [B,D] or [B,L,D]; use final causal row for tabular comparison.
        if x.dim()==3: x=x[:,-1]
        z=self.encoder(self.proj(x).unsqueeze(1)).squeeze(1)
        return self.cls(z)

class GRUBaseline(nn.Module):
    def __init__(self,d,h=96):
        super().__init__(); self.gru=nn.GRU(d,h,batch_first=True); self.cls=nn.Linear(h,4)
    def forward(self,x):
        z,_=self.gru(x); return self.cls(z[:,-1])

import numpy as np
import torch
from torch import nn
from sklearn.metrics import log_loss

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__(); self.log_temperature=nn.Parameter(torch.zeros(()))
    @property
    def temperature(self): return self.log_temperature.exp().clamp(0.05,20.0)
    def forward(self, logits): return logits/self.temperature
    def fit(self, logits, labels, max_iter=100):
        logits=torch.as_tensor(logits,dtype=torch.float32); labels=torch.as_tensor(labels,dtype=torch.long)
        opt=torch.optim.LBFGS([self.log_temperature],lr=0.1,max_iter=max_iter,line_search_fn="strong_wolfe")
        loss_fn=nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss=loss_fn(self.forward(logits),labels); loss.backward(); return loss
        opt.step(closure); return self
    def predict_proba(self, logits): return torch.softmax(self.forward(torch.as_tensor(logits,dtype=torch.float32)),-1).detach().numpy()

def multiclass_brier(y, p):
    y=np.asarray(y); p=np.asarray(p)
    one=np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p-one)**2,axis=1)))

def calibration_metrics(y,p):
    b=multiclass_brier(y,p); ll=float(log_loss(y,p,labels=list(range(p.shape[1]))))
    # Higher is better, bounded below at zero. Brier max for 4 classes is 2.
    score=float(np.clip(1.0-b/2.0,0,1))
    return {"brier":b,"log_loss":ll,"calibration_score":score}


import math
import torch
from torch import nn
import torch.nn.functional as F
from heads.multitask import MultiTaskHeads

class BaseMT(nn.Module):
    def __init__(self,d,h=128):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.heads=MultiTaskHeads(h)
    def encode(self,x): return self.proj(x[:,-1])
    def forward(self,x,stage_target=None):
        z=self.encode(x)
        return self.heads(z,stage_target=stage_target)

class TGN(BaseMT):
    """Lightweight event-memory TGN-style implementation."""
    def __init__(self,d,h=128,nodes=3):
        super().__init__(d,h); self.mem=nn.GRUCell(h,h); self.node_mem=nn.Parameter(torch.zeros(nodes,h))
        self.msg=nn.Linear(d,h); self.nodes=nodes
    def encode(self,x):
        B,L,C=x.shape
        mem=self.node_mem.unsqueeze(0).expand(B,-1,-1)
        for t in range(L):
            m=self.msg(x[:,t]).unsqueeze(1).expand(-1,self.nodes,-1)
            mem=self.mem(m.reshape(-1,mem.size(-1)),mem.reshape(-1,mem.size(-1))).reshape(B,self.nodes,-1)
        return mem.mean(1)

class GraphConv(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.lin=nn.Linear(d,h)
    def forward(self,x,A):
        return torch.relu(self.lin(torch.einsum("ij,bljd->blid",A,x)))

class EvolveGCN(BaseMT):
    def __init__(self,d,h=128,nodes=3):
        super().__init__(d,h); self.gconv=GraphConv(10,h); self.rnn=nn.GRU(h,h,batch_first=True); self.nodes=nodes
    def encode(self,x):
        # x is flattened sequence; reshape the first 30 node channels if available.
        C=x.size(-1); nf=min(10,C//3); usable=nf*3
        nodes=x[:,:,:usable].reshape(x.size(0),x.size(1),3,nf)
        A=torch.ones(3,3,device=x.device)/3
        g=self.gconv(nodes,A).mean(2)
        z,_=self.rnn(g)
        return z[:,-1]

class GraphWaveNet(BaseMT):
    def __init__(self,d,h=128,nodes=3):
        super().__init__(d,h); self.temporal=nn.Conv1d(d,h,kernel_size=3,padding=2,dilation=1); self.adaptive=nn.Parameter(torch.randn(nodes,nodes)*.05)
    def encode(self,x):
        z = torch.relu(self.temporal(x.transpose(1,2)))[:, :, :x.size(1)]
        # Adaptive support is used as a lightweight feature-mixing gate when the
        # supplied data do not expose a physical adjacency matrix.
        support = torch.softmax(self.adaptive, dim=-1)
        if support.size(0) == z.size(1):
            z = z * (1.0 + support.mean(-1).view(1, -1, 1))
        return z[:, :, -1]

class TFT(BaseMT):
    def __init__(self,d,h=128):
        super().__init__(d,h); self.lstm=nn.LSTM(d,h,batch_first=True); self.attn=nn.MultiheadAttention(h,4,batch_first=True)
        self.gate=nn.Sequential(nn.Linear(h,h),nn.Sigmoid())
    def encode(self,x):
        z,_=self.lstm(x); a,_=self.attn(z,z,z); return (self.gate(a[:,-1])*z[:,-1])

class TimesNet(BaseMT):
    def __init__(self,d,h=128):
        super().__init__(d,h); self.conv2=nn.Conv2d(1,h,kernel_size=(3,3),padding=1); self.pool=nn.AdaptiveAvgPool2d((1,1))
    def encode(self,x):
        # Treat [time, channel] as a 2-D variation map.
        z=torch.relu(self.conv2(x.unsqueeze(1))); return self.pool(z).flatten(1)

class PatchTST(BaseMT):
    def __init__(self,d,h=128,patch=8):
        super().__init__(d,h); self.patch=patch; self.embed=nn.Linear(d*patch,h); enc=nn.TransformerEncoderLayer(h,4,batch_first=True)
        self.tr=nn.TransformerEncoder(enc,2)
    def encode(self,x):
        L=x.size(1); n=L//self.patch
        x=x[:,:n*self.patch].reshape(x.size(0),n,self.patch,x.size(2)).flatten(2)
        z=self.tr(self.embed(x)); return z[:,-1]

class NeuralCDE(BaseMT):
    """Neural-CDE-inspired Euler controlled differential equation fallback.
    Timestamps/missingness can be supplied later; regular data uses causal Euler updates."""
    def __init__(self,d,h=128):
        super().__init__(d,h); self.f=nn.Sequential(nn.Linear(h+d,h),nn.Tanh(),nn.Linear(h,h))
    def encode(self,x):
        z=torch.zeros(x.size(0),self.proj.out_features,device=x.device)
        for t in range(x.size(1)):
            z=z+0.05*self.f(torch.cat([z,x[:,t]],-1))
        return z

class DeepSSM(BaseMT):
    def __init__(self,d,h=128):
        super().__init__(d,h); self.gru=nn.GRU(d,h,batch_first=True); self.emit=nn.Linear(h,h)
    def encode(self,x):
        z,_=self.gru(x); return torch.tanh(self.emit(z[:,-1]))

class Mamba(BaseMT):
    """Dependency-light selective state-space implementation matching the causal idea."""
    def __init__(self,d,h=128):
        super().__init__(d,h); self.inp=nn.Linear(d,h); self.delta=nn.Linear(d,h); self.B=nn.Linear(d,h); self.C=nn.Linear(d,h)
    def encode(self,x):
        z=torch.zeros(x.size(0),self.proj.out_features,device=x.device)
        for t in range(x.size(1)):
            dt=torch.sigmoid(self.delta(x[:,t]))
            z=(1-dt)*z+dt*torch.tanh(self.B(x[:,t]))*torch.sigmoid(self.C(x[:,t]))
        return z

class DeepHit(BaseMT):
    def __init__(self,d,h=128,horizon=64):
        super().__init__(d,h); self.encoder=nn.GRU(d,h,batch_first=True); self.event_pmf=nn.Linear(h,4*horizon); self.horizon=horizon
    def encode(self,x): 
        z,_=self.encoder(x); return z[:,-1]
    def forward(self,x,stage_target=None):
        z=self.encode(x); out=self.heads(z,stage_target=stage_target); out["event_pmf"]=self.event_pmf(z).reshape(x.size(0),4,self.horizon)
        return out

MODEL_REGISTRY={
    "tgn":TGN,"evolvegcn":EvolveGCN,"graph_wavenet":GraphWaveNet,"tft":TFT,
    "timesnet":TimesNet,"patchtst":PatchTST,"neural_cde":NeuralCDE,
    "deep_ssm":DeepSSM,"mamba":Mamba,"deephit":DeepHit
}

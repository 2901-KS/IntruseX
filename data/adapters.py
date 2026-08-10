import torch
from torch import nn

class SequenceAdapter(nn.Module):
    def forward(self,x):
        return x

class NodeGroupGraphAdapter(nn.Module):
    """
    Graph view derived only from the supplied H/G/I telemetry groups.
    These are telemetry entities, not fabricated physical edge weights.
    """
    def __init__(self, columns):
        super().__init__()
        self.columns=list(columns)
        self.groups=["H","G","I"]
        self.node_cols={g:[i for i,c in enumerate(columns) if c.endswith("_"+g)] for g in self.groups}
        self.register_buffer("support", torch.tensor([
            [1.0,1.0,1.0],
            [1.0,1.0,1.0],
            [1.0,1.0,1.0],
        ]) / 3.0)
    def forward(self,x):
        nodes=[x[:,:,ix] for ix in self.node_cols.values()]
        return torch.stack(nodes,dim=2), self.support

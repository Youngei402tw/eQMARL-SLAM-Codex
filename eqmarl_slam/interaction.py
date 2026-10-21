"""Two-robot symmetric local-plus-edge component model (not a task policy)."""
import torch
from torch import nn
from .critics import SplitClassical

class InteractionCritic(SplitClassical):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'bilinear':
            self.edge = nn.Parameter(torch.randn(2,2,dtype=torch.float64)*.1)
        elif kind in ['quantum_noent','eqmarl']:
            from .quantum import QuantumCircuit
            self.theta = nn.Parameter(torch.rand(2,2,2,3,dtype=torch.float64)*torch.pi)
            self.scales = nn.Parameter(torch.ones(2,1,2,1,dtype=torch.float64))
            self.head = nn.Linear(1,1,dtype=torch.float64)
            self.circuit = QuantumCircuit(2,2,1,entanglement=kind=='eqmarl')
        elif kind != 'none':
            raise ValueError(kind)

    def forward(self,x,delivered=None):
        z = self.encoder(x)
        value = self.local(z).sum(-2).squeeze(-1)
        if self.kind=='none':return value
        if self.kind=='bilinear':
            pair = torch.einsum('...i,ij,...j->...',z[...,0,:],(self.edge+self.edge.T)/2,z[...,1,:])
        else:
            pair = self.head((self.circuit(z,self.theta,self.scales)+self.circuit(z.flip(-2),self.theta,self.scales))/2).squeeze(-1)
        return value+pair*(1 if delivered is None else delivered)

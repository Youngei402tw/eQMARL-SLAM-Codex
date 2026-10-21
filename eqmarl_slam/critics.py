"""Matched local compression and explicit critic-only interventions."""
from contextlib import contextmanager

import torch
from torch import nn

from .mappo import ClassicalCritic

METHODS = ["split_classical","central_classical","bilinear","quantum_noent","eqmarl","qfctde"]


class SplitClassical(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(27,2,dtype=torch.float64)
        self.local = nn.Sequential(nn.Linear(2,16,dtype=torch.float64),nn.Tanh(),nn.Linear(16,1,dtype=torch.float64))

    def forward(self, x):
        return self.local(self.encoder(x)).sum(-2).squeeze(-1)


class Bilinear(SplitClassical):
    def __init__(self):
        super().__init__()
        self.interaction = nn.Parameter(torch.randn(2,2,dtype=torch.float64)*.1)

    def forward(self, x):
        z = self.encoder(x)
        return self.local(z).sum(-2).squeeze(-1)+torch.einsum('...i,ij,...j->...',z[...,0,:],self.interaction,z[...,1,:])


class QuantumValue(nn.Module):
    def __init__(self, entangled=True, qf=False):
        super().__init__()
        from .quantum import QuantumCircuit
        self.encoder = nn.Linear(27,2,dtype=torch.float64)
        self.qf = qf
        parts,d = (1,4) if qf else (2,2)
        self.theta = nn.Parameter(torch.rand(parts,2,d,3,dtype=torch.float64)*torch.pi)
        self.scales = nn.Parameter(torch.ones(parts,1,d,1,dtype=torch.float64))
        self.head = nn.Linear(1,1,dtype=torch.float64)
        self.circuit = QuantumCircuit(parts,d,1,entanglement=entangled)

    def forward(self, x):
        z = self.encoder(x)
        if self.qf: z = z.reshape(*z.shape[:-2],1,4)
        return self.head(self.circuit(z,self.theta,self.scales)).squeeze(-1)


def make_critic(method, seed):
    torch.manual_seed(seed)
    if method=='split_classical': return SplitClassical()
    if method=='central_classical': return ClassicalCritic()
    if method=='bilinear': return Bilinear()
    if method=='quantum_noent': return QuantumValue(False)
    if method=='eqmarl': return QuantumValue(True)
    if method=='qfctde': return QuantumValue(True,True)
    raise ValueError(f'Unknown critic {method}')


@contextmanager
def tracked(critic, resources):
    if not hasattr(critic,'circuit'):
        yield
        return
    import pennylane as qml
    monitor = qml.Tracker(critic.circuit.device)
    try:
        with monitor: yield
    finally:
        resources['circuit_evaluations'] = resources.get('circuit_evaluations',0)+int(monitor.totals.get('executions',0))
        resources['shots'] = resources.get('shots',0)+int(monitor.totals.get('shots',0))


def train_critic(config, output, emit, progress=None):
    from .mappo import train
    critic = make_critic(config['critic_method'],config['critic_seed'])
    counters = {'circuit_evaluations':0,'shots':0}
    result = None
    try:
        with tracked(critic,counters):
            result = train(config,output,emit,critic=critic,progress=progress)
    finally:
        if result is not None: result['resources'].update(counters)
        elif progress is not None: progress.setdefault('resources',{}).update(counters)
    result['summary']['critic_method'] = config['critic_method']
    result['summary']['critic_parameters'] = sum(p.numel() for p in critic.parameters())
    if hasattr(critic,'circuit'): result['summary']['quantum_resources']=critic.circuit.resource_report()
    return result

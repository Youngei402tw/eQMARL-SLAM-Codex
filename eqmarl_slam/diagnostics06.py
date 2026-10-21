"""Declared synthetic mechanism attribution plus real frozen-return fitting.

Synthetic conditions do not claim environmental interventions or policy gains.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .interaction import InteractionCritic
from .critics import tracked


def target(x,condition):
    a,b=x[...,0],x[...,1]
    y=.2*a.sum(-1)
    if condition in ['uncertainty','all']:
        y=y+.1*b.square().sum(-1)
    if condition in ['complementarity','limited_communication','all']:
        edge=.4*(a[:,0]*b[:,1]+a[:,1]*b[:,0])
        if condition in ['limited_communication','all']:edge=edge*(x[:,0,2]>0).double()
        y=y+edge
    if condition in ['rendezvous','all']:
        y=y+.3*torch.exp(-((x[:,0,:2]-x[:,1,:2])**2).sum(-1))
    return y


def run_diagnostics(config,output,emit,progress=None):
    torch.set_num_threads(1)
    output=Path(output)
    resources={'primitive_steps':0,'decisions':0,'updates':0,'circuit_evaluations':0,'shots':0}
    if progress is not None:progress['resources']=resources
    rng=np.random.default_rng(config['data_seed'])
    # First three slots carry synthetic utility coordinates; others zero.
    x=torch.zeros((config['train_rows']+config['validation_rows'],2,27),dtype=torch.float64)
    x[:,:,:3]=torch.tensor(rng.uniform(-1,1,(len(x),2,3)),dtype=torch.float64)
    # Delivery is an undirected edge event, the same at both endpoints.
    x[:,1,2]=x[:,0,2]
    datasets={c:(x,target(x,c), (x[:,0,2]>0).double() if c in ['limited_communication','all'] else torch.ones(len(x),dtype=torch.float64)) for c in config['conditions']}
    real=json.loads((Path(__file__).resolve().parents[1]/config['return_dataset']).read_text())['dataset']
    ordered=[r for r in real if r['split']=='train']+[r for r in real if r['split']=='validation']
    rx=torch.tensor([r['observations'] for r in ordered],dtype=torch.float64)
    ry=torch.tensor([r['return_target']/10 for r in ordered],dtype=torch.float64)
    datasets['real_returns']=(rx,ry,torch.ones(len(rx),dtype=torch.float64))
    corpus={name:{'observations':xx.tolist(),'targets':yy.tolist(),'delivered':dd.tolist(),'train_rows':config['train_rows'] if name!='real_returns' else sum(r['split']=='train' for r in ordered)} for name,(xx,yy,dd) in datasets.items()}
    (output/'interaction_dataset.json').write_text(json.dumps(corpus,indent=2)+'\n')
    gates={};results=[]
    for condition,(xx,yy,delivered) in datasets.items():
        n=corpus[condition]['train_rows']
        for seed in config['model_seeds']:
            for kind in config['methods']:
                torch.manual_seed(seed);model=InteractionCritic(kind)
                optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
                norms={}
                with tracked(model,resources):
                    for step in range(config['updates']):
                        prediction=model(xx[:n],delivered[:n]);loss=(prediction-yy[:n]).square().mean()
                        optimizer.zero_grad();loss.backward()
                        norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite component fit')
                        optimizer.step();resources['updates']+=1
                        emit({'kind':'interaction_update','condition':condition,'method':kind,'seed':seed,'update':step+1,'mse':float(loss.detach()),'gradient_norms':norms})
                    with torch.no_grad():
                        values=model(xx,delivered);swapped=model(xx.flip(-2),delivered)
                        swap=float((values-swapped).abs().max())
                        train_mse=float((values[:n]-yy[:n]).square().mean());val_mse=float((values[n:]-yy[n:]).square().mean())
                gates[f'{condition}-{kind}-{seed}-finite_symmetric']=bool(np.isfinite(val_mse) and swap<1e-10 and norms.get('encoder.weight',0)>1e-12)
                row=dict(condition=condition,method=kind,seed=seed,training_mse=train_mse,validation_mse=val_mse,swap_error=swap,parameters=sum(p.numel() for p in model.parameters()),predictions=values.tolist(),targets=yy.tolist())
                emit({'kind':'interaction_result',**row});results.append(row)
                torch.save({'model':model.state_dict(),'kind':kind,'seed':seed,'condition':condition},output/f'{condition}-{kind}-{seed}.pt')
    return {'passed':all(gates.values()),'gates':gates,'resources':resources,'summary':{'results':results,'scope':'Synthetic utility interventions and frozen real-return component fitting only; no policy retraining or environmental causal claim.','communication':'Training edge needs both local2-float embeddings and two unary values; float64 payload48 bytes per undirected edge evaluation before transport overhead; zero edge executions when no edge is modeled. Symmetry averaging doubles quantum calls. Runtime deployed actor remains classical.'}}

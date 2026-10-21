"""Frozen-actor return corpus, critic fitting, and signed-capacity checks."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from .critics import METHODS, make_critic, tracked
from .mappo import ActorCritic, collect_episode


def run_diagnostics(config,output,emit,progress=None):
    torch.set_num_threads(1)
    resources={'primitive_steps':0,'decisions':0,'updates':0,'circuit_evaluations':0,'shots':0}
    if progress is not None: progress['resources']=resources
    dataset=[]
    for seed in config['frozen_actor_seeds']:
        path=Path(__file__).resolve().parents[1]/config['actor_checkpoints'][str(seed)]
        model=ActorCritic()
        payload=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(payload['model'])
        actor_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        generator=torch.Generator().manual_seed(config['dataset_action_seed']+seed)
        for split in ['train','validation']:
            for scenario in config['dataset_scenarios'][split]:
                def charge(duration): resources['primitive_steps']+=duration
                def decision(): resources['decisions']+=2
                rows,metrics=collect_episode(model,config['task'],scenario,generator,primitive_limit=config['task']['horizon'],
                                             training=True,on_transition=charge,on_decision=decision)
                target=0.
                for row in reversed(rows):
                    target=row['reward']+config['task']['gamma']**row['duration']*target
                    dataset.append({'split':split,'observations':row['obs'].tolist(),'return_target':target,
                                    'actor_seed':seed,'actor_checkpoint_sha256':actor_hash,'scenario_id':scenario['scenario_id']})
                emit({'kind':'frozen_actor_episode','split':split,'actor_seed':seed,'optimizer_updates':0,**metrics})
    output=Path(output)
    (output/'return_dataset.json').write_text(json.dumps({'dataset':dataset},indent=2)+'\n')
    arrays={}
    for split in ['train','validation']:
        rows=[r for r in dataset if r['split']==split]
        arrays[split]=(torch.tensor([r['observations'] for r in rows],dtype=torch.float64),
                       torch.tensor([r['return_target']/10 for r in rows],dtype=torch.float64))
    gates, summaries={},[]
    started=time.perf_counter()
    for method in config['methods']:
        if time.perf_counter()-started>config['wall_seconds']: raise TimeoutError('Critic fit family cap')
        critic=make_critic(method,config['critic_seed'])
        optimizer=torch.optim.Adam(critic.parameters(),lr=config['fit_learning_rate'])
        x,y=arrays['train']
        with tracked(critic,resources):
            with torch.no_grad(): initial=float((critic(x)-y).square().mean())
            gradient_groups={}
            for update in range(config['fit_updates']):
                prediction=critic(x);loss=(prediction-y).square().mean()
                optimizer.zero_grad();loss.backward()
                gradient_groups={name:float(parameter.grad.norm()) if parameter.grad is not None else None for name,parameter in critic.named_parameters()}
                if not torch.isfinite(loss): raise FloatingPointError(f'{method} nonfinite fitting loss')
                optimizer.step();resources['updates']+=1
                emit({'kind':'value_fit_update','method':method,'update':update+1,'normalized_mse':float(loss.detach()),
                      'gradient_norms':gradient_groups})
            with torch.no_grad():
                train_prediction=critic(x);validation_prediction=critic(arrays['validation'][0])
                training_mse=float((train_prediction-y).square().mean())
                validation_mse=float((validation_prediction-arrays['validation'][1]).square().mean())
            emit({'kind':'fit_predictions','method':method,'train_targets':y.tolist(),'train_predictions':train_prediction.tolist(),
                  'validation_targets':arrays['validation'][1].tolist(),'validation_predictions':validation_prediction.tolist()})
        torch.save({'method':method,'model':critic.state_dict(),'return_scale':10.,'critic_seed':config['critic_seed']},output/f'fitted-{method}.pt')
        # Fresh model: fitting negative and positive values is tested separately
        # from the real return distribution and does not change the RL critic.
        capacity=make_critic(method,config['critic_seed']+1)
        opt=torch.optim.Adam(capacity.parameters(),lr=config['capacity_learning_rate'])
        cx=torch.tensor(np.random.default_rng(config['capacity_seed']).uniform(-1,1,(2,2,27)),dtype=torch.float64)
        cy=torch.tensor([-.1,.1],dtype=torch.float64)
        with tracked(capacity,resources):
            for _ in range(config['capacity_updates']):
                error=(capacity(cx)-cy).square().mean();opt.zero_grad();error.backward();opt.step();resources['updates']+=1
            with torch.no_grad(): capacity_prediction=capacity(cx);capacity_mse=float((capacity_prediction-cy).square().mean())
        connected=all(value is not None and np.isfinite(value) for value in gradient_groups.values())
        # A final Z rotation legitimately commutes with global parity; some
        # individual coordinates can vanish. Each intended group must connect.
        connected=connected and any(value>1e-12 for name,value in gradient_groups.items() if name.startswith('encoder'))
        if hasattr(critic,'circuit'):connected=connected and gradient_groups.get('theta',0)>1e-12
        gates[f'{method}_gradients']=bool(connected)
        gates[f'{method}_signed_capacity']=bool(capacity_mse<config['capacity_mse_threshold'])
        row={'method':method,'initial_training_mse':initial,'training_mse':training_mse,'validation_mse':validation_mse,
             'capacity_mse':capacity_mse,'capacity_predictions':capacity_prediction.tolist(),
             'parameters':sum(p.numel() for p in critic.parameters()),'gradients_connected':connected}
        summaries.append(row);emit({'kind':'critic_fit_result',**row})
    q1,q0=make_critic('eqmarl',config['critic_seed']),make_critic('quantum_noent',config['critic_seed'])
    gates['matched_entanglement_only']=all(torch.equal(value,q0.state_dict()[name]) for name,value in q1.state_dict().items())
    return {'passed':all(gates.values()),'gates':gates,'resources':resources,
            'summary':{'fits':summaries,'dataset_rows':len(dataset),'target_sign_range':{split:[float(y.min()),float(y.max())] for split,(x,y) in arrays.items()},
                       'actor_condition':'all three archived pilot actors equally represented; frozen corpus collection only',
                       'fit_interpretation':'Real return underfitting is descriptive; signed-capacity/gradient correctness gate required before RL.',
                       'qfctde':'Additional global four-wire topology, not matched no-entanglement ablation'}}

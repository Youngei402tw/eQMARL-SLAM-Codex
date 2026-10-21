"""Frozen-critic sensitivity; no actor training or finite-shot policy claim."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .critics import make_critic,tracked
from .quantum import QuantumCircuit


def run_diagnostics(config,output,emit,progress=None):
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];output=Path(output)
    resources={'primitive_steps':0,'decisions':0,'updates':0,'circuit_evaluations':0,'shots':0}
    if progress is not None:progress['resources']=resources
    rows=json.loads((root/config['dataset']).read_text())['dataset']
    selected=[r for r in rows if r['split']=='validation'][:config['rows']]
    x=torch.tensor([r['observations'] for r in selected],dtype=torch.float64)
    gates={};results=[];provenance=[]
    def record(row):
        if resources['shots']>config['max_shots'] or resources['circuit_evaluations']>config['max_circuit_evaluations']:
            raise TimeoutError('Registered shot/circuit cap reached')
        emit(row)
    for method in config['methods']:
        checkpoint=root/config['checkpoints'][method]
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        provenance.append({'method':method,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'optimizer_updates':0})
        def fresh(shots=None,noise=None,seed=None):
            model=make_critic(method,config['critic_seed']);model.load_state_dict(payload['model'])
            model.circuit=QuantumCircuit(2,2,1,entanglement=method=='eqmarl',shots=shots,noise=noise,seed=seed)
            return model
        exact=fresh()
        with tracked(exact,resources):
            value=exact(x);value.sum().backward()
            reference_gradient=torch.cat([p.grad.flatten() for p in exact.parameters()]).detach()
        reference=value.detach()
        record({'kind':'exact_reference','method':method,'normalized_values':reference.tolist(),'gradient':reference_gradient.tolist()})
        # Same device seed must replay the finite-shot sequence exactly.
        a,b=fresh(64,seed=config['shot_seed']),fresh(64,seed=config['shot_seed'])
        with tracked(a,resources),tracked(b,resources),torch.no_grad():
            gates[method+'-shot_seed_replay']=bool(torch.equal(a(x),b(x)))
        for shots in config['shot_settings']:
            observations=[];gradients=[]
            for repeat in range(config['repetitions']):
                model=fresh(shots,seed=config['shot_seed']+10000*shots+repeat)
                with tracked(model,resources):
                    val=model(x)
                    # Differentiate one input to bound cost; includes encoder,
                    # theta, scale and affine head, using valid parameter shift.
                    one=model(x[:1]);one.sum().backward()
                    gradient=torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
                observations.append(val.detach().tolist());gradients.append(gradient.tolist())
                record({'kind':'shot_replicate','method':method,'shots_per_execution':shots,'repeat':repeat,'normalized_values':val.detach().tolist(),'gradient':gradient.tolist(),'cumulative_resources':dict(resources)})
            # Compare gradients against the same one-observation analytic loss.
            single=fresh()
            with tracked(single,resources):single(x[:1]).sum().backward()
            g=torch.cat([p.grad.flatten() for p in single.parameters()]).detach().numpy()
            values=np.asarray(observations);gradient=np.asarray(gradients)
            mask=[];group_errors={};offset=0
            for name,parameter in single.named_parameters():
                size=parameter.numel();segment=slice(offset,offset+size);offset+=size
                group_errors[name]={'reference_norm':float(np.linalg.norm(g[segment])),'absolute_error':float(np.linalg.norm(gradient.mean(0)[segment]-g[segment])),'relative_error':float(np.linalg.norm(gradient.mean(0)[segment]-g[segment])/max(np.linalg.norm(g[segment]),1e-12))}
                mask.extend([not name.startswith('head.')] * size)
            mask=np.asarray(mask,dtype=bool)
            record({'kind':'analytic_single_gradient','method':method,'shots_condition':shots,'gradient':g.tolist(),'parameter_groups':[{'name':name,'size':parameter.numel()} for name,parameter in single.named_parameters()]})
            row={'method':method,'shots':shots,'normalized_value_rmse':float(np.sqrt(np.mean((values-reference.numpy())**2))),'physical_value_rmse':float(10*np.sqrt(np.mean((values-reference.numpy())**2))),'gradient_relative_rmse':float(np.linalg.norm((gradient.mean(0)-g)[mask])/max(np.linalg.norm(g[mask]),1e-12)),'all_parameter_gradient_relative_rmse':float(np.linalg.norm(gradient.mean(0)-g)/max(np.linalg.norm(g),1e-12)),'gradient_group_errors':group_errors,'gradient_reference_norm':float(np.linalg.norm(g[mask])),'empirical_value_variance':values.var(0,ddof=1).tolist()}
            gates[f'{method}-{shots}-finite_parameter_shift']=bool(np.isfinite(gradient).all() and np.isfinite(values).all())
            results.append(row);record({'kind':'shot_result',**row})
        for noise in config['noise_probabilities']:
            model=fresh(noise=noise)
            with tracked(model,resources),torch.no_grad():noisy=model(x)
            # A final independent per-wire depolarizing channel scales global
            # four-wire parity by (1-4p/3)^4 before the affine head.
            factor=(1-4*noise/3)**4
            bias=model.head.bias.detach().squeeze()
            expected=(reference-bias)*factor+bias
            residual=float((noisy-expected).abs().max())
            gates[f'{method}-noise-{noise}-channel_identity']=residual<1e-8
            samples=[]
            for repeat in range(config['repetitions']):
                sampled=fresh(config['noise_shots'],noise,config['shot_seed']+200000+int(noise*10000)+repeat)
                with tracked(sampled,resources),torch.no_grad():v=sampled(x)
                samples.append(v.tolist())
                record({'kind':'noise_replicate','method':method,'depolarizing_probability':noise,'shots_per_execution':config['noise_shots'],'repeat':repeat,'normalized_values':v.tolist(),'cumulative_resources':dict(resources)})
            row={'method':method,'depolarizing_probability':noise,'analytic_normalized_values':noisy.tolist(),'channel_identity_error':residual,'normalized_bias_rmse':float(torch.sqrt((noisy-reference).square().mean())),'sample_rmse_vs_noisy_expectation':float(np.sqrt(np.mean((np.asarray(samples)-noisy.numpy())**2)))}
            results.append(row);record({'kind':'noise_result',**row})
    return {'passed':all(gates.values()),'gates':gates,'resources':resources,'summary':{'results':results,'checkpoint_provenance':provenance,'scope':'Frozen analytic-trained critic sensitivity on validation inputs; no finite-shot/noisy policy trained, no task-performance or hardware/network claim.','noise_model':'One independent depolarizing channel per wire after all unitary gates, before global parity readout; exact default.mixed expectations and finite-shot default.mixed kept separate.','gradient_reliability_threshold':config['gradient_relative_error_threshold'],'unreliable_settings':[r for r in results if r.get('gradient_relative_rmse',0)>config['gradient_relative_error_threshold']]}}

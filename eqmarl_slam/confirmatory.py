"""Locked-budget training followed by frozen, actor-only paired testing."""
import hashlib,json,time,resource
from pathlib import Path
import torch
from .belief_task import BeliefTask
from .critics import make_critic,train_critic
from .mappo import ActorCritic


from .policy_evaluation import evaluate_actor


ROOT = Path(__file__).resolve().parents[1]


def resolve_test_scenarios(config):
    """Load a locked manifest only at execution time, after source publication."""
    if "test_scenarios" in config:  # Retain historical Stage 08 reproduction.
        return config["test_scenarios"], {"mode": "embedded_historical", "count": len(config["test_scenarios"])}
    reference = config["test_scenarios_manifest"]
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("Locked test manifest path or hash mismatch")
    payload = json.loads(path.read_text())
    scenarios = payload["scenarios"]
    if payload.get("split") != "test" or len(scenarios) != reference["expected_count"]:
        raise ValueError("Locked test manifest split/count mismatch")
    return scenarios, {"mode": "deferred_locked_manifest", "path": reference["path"],
                       "sha256": reference["sha256"], "count": len(scenarios)}


def train_and_test(config,output,emit,progress=None):
    result=train_critic(config,output,emit,progress=progress)
    resources=result['resources']
    if progress is not None:progress['resources']=resources
    checkpoint=Path(output)/'final.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Reload the serialized final policy; the unused critic is never executed.
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=ActorCritic(critic=make_critic(config['critic_method'],config['critic_seed']),actor_seed=config['actor_seed'],critic_seed=config['critic_seed'],hidden=config['actor_hidden'])
    model.load_state_dict(payload['model']);model.eval()
    initial={k:v.clone() for k,v in model.actor.state_dict().items()}
    cpu=lambda: sum(getattr(resource.getrusage(who),field) for who in [resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN] for field in ["ru_utime","ru_stime"])
    test_scenarios,test_source=resolve_test_scenarios(config)
    tests=[];test_start=time.perf_counter();test_cpu_start=cpu()
    for scenario in test_scenarios:
        row=evaluate_actor(model.actor,config['task'],scenario,resources)
        tests.append(row);emit({'kind':'test_episode','checkpoint_sha256':before,**row})
    after=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result['gates']['frozen_test_policy']=before==after and all(torch.equal(v,initial[k]) for k,v in model.actor.state_dict().items())
    result['gates']['complete_locked_test']=len(tests)==len(test_scenarios)
    result['passed']=all(result['gates'].values())
    result['summary'].update(test=tests,test_checkpoint_sha256=before,test_wall_seconds=time.perf_counter()-test_start,test_cpu_core_hours=(cpu()-test_cpu_start)/3600,test_critic_executions=0,test_optimizer_updates=0,locked_analysis_plan=config['analysis_plan'],test_scenario_source=test_source)
    return result

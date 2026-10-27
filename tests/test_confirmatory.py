import numpy as np
import torch
import hashlib,json
from eqmarl_slam.belief_task import DEFAULT_TASK
from eqmarl_slam.critics import make_critic
from eqmarl_slam.mappo import ActorCritic,collect_episode
from eqmarl_slam.policy_evaluation import evaluate_actor
import eqmarl_slam.confirmatory as confirmatory


def test_paired_actor_initialization_is_independent_of_critic_choice():
    actors=[]
    for method in ['central_classical','quantum_noent','eqmarl']:
        model=ActorCritic(critic=make_critic(method,9801),actor_seed=8801,critic_seed=9801)
        actors.append(model.actor.state_dict())
    assert all(torch.equal(v,actor[k]) for actor in actors[1:] for k,v in actors[0].items())


def test_actor_only_evaluation_matches_existing_greedy_rollout():
    model=ActorCritic(actor_seed=8001,critic_seed=8002)
    task={**DEFAULT_TASK,'horizon':4}
    scenario={'scenario_id':'preflight-actor-only-8001','map_seed':8001,'noise_seed':8002}
    _,expected=collect_episode(model,task,scenario,torch.Generator().manual_seed(8003),primitive_limit=4,training=False)
    resources={'primitive_steps':0,'decisions':0}
    actual=evaluate_actor(model.actor,task,scenario,resources)
    for key in ['physical_coverage','fixed_landmark_accuracy','pose_error_m','trajectory_ate_m','return','communication_bytes']:
        assert np.isclose(actual[key],expected[key],atol=1e-10,rtol=0)
    assert sum(map(sum,actual['action_counts_by_agent']))==resources['decisions']
    assert np.isclose(sum(actual['reward_component_totals'].values()),actual['return'],atol=1e-10)


def test_test_manifest_is_deferred_and_hash_locked(tmp_path,monkeypatch):
    payload={"split":"test","scenarios":[{"scenario_id":"fixture","map_seed":1,"noise_seed":2}]}
    path=tmp_path/"test.json"; path.write_text(json.dumps(payload))
    monkeypatch.setattr(confirmatory,"ROOT",tmp_path)
    config={"test_scenarios_manifest":{"path":"test.json",
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"expected_count":1}}
    scenarios,source=confirmatory.resolve_test_scenarios(config)
    assert len(scenarios)==1
    assert source["mode"]=="deferred_locked_manifest"

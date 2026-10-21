"""Replay frozen pilot policies; original run bundles remain immutable."""
import hashlib
import json
from pathlib import Path
import torch
from .classical import SharedGRUActor
from .policy_evaluation import evaluate_actor


def run_diagnostics(config, output, emit, progress=None):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    resources = {'primitive_steps':0,'decisions':0,'updates':0,'circuit_evaluations':0,'shots':0}
    if progress is not None: progress['resources'] = resources
    gates, results = {}, []
    for run in config['pilot_runs']:
        folder = root/'experiments/05'/run
        cfg = json.loads((folder/'resolved_config.yaml').read_text())
        checkpoint = folder/'final.pt'
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        actor = SharedGRUActor(27,cfg['actor_hidden'],5).double()
        actor.load_state_dict({k.removeprefix('actor.'):v for k,v in payload['model'].items() if k.startswith('actor.')})
        actor.eval()
        scenario = cfg['validation_scenarios'][0]
        def macro(row): emit({'kind':'frozen_replay_macro','pilot_run':run,**row})
        actual = evaluate_actor(actor,cfg['task'],scenario,resources,emit_macro=macro)
        original = next(json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()
                        if json.loads(line)['kind']=='validation_episode' and json.loads(line)['scenario_id']==scenario['scenario_id'])
        fields = ['physical_coverage','estimated_coverage','fixed_landmark_accuracy','pose_error_m',
                  'trajectory_ate_m','trajectory_rpe_m','travel_distance_m','collisions',
                  'communication_bytes','messages','success','return','primitive_steps']
        errors = {k:abs(float(actual[k])-float(original[k])) for k in fields}
        gates[run+'-frozen_replay'] = max(errors.values())<config['replay_atol'] and digest==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        row = {'pilot_run':run,'method':cfg['critic_method'],'seed':cfg['actor_seed'],
               'checkpoint_sha256':digest,'absolute_errors':errors,**actual}
        emit({'kind':'frozen_replay_episode',**row})
        results.append(row)
    return {'passed':all(gates.values()),'gates':gates,'resources':resources,
            'summary':{'results':results,'scope':'One fixed validation-map frozen replay per pilot; action/reward counts describe this replay, not unrecorded original training trajectories. No optimizer updates or critic executions.'}}

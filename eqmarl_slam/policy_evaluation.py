"""Frozen actor-only evaluation with action, duration and reward diagnostics."""
import numpy as np
import torch
from .belief_task import BeliefTask


def evaluate_actor(actor, task_config, scenario, resources, emit_macro=None):
    env = BeliefTask(task_config, scenario)
    obs, mask = env.reset()
    hidden = actor.initial_state(agents=2)
    def charge(n): resources['primitive_steps'] += n
    env.on_primitive = charge
    total, macros, done = 0., 0, False
    actions_by_agent = np.zeros((2,5), dtype=int)
    durations = {1:0, 2:0, 3:0}
    parts = {k:0. for k in ['coverage','information','travel','collision','communication','coordination']}
    try:
        while not done:
            with torch.no_grad():
                distribution, hidden = actor(torch.tensor(obs,dtype=torch.float64), hidden,
                                             torch.tensor(mask,dtype=torch.bool))
                action = distribution.probs.argmax(-1).numpy()
            resources['decisions'] += 2
            macros += 1
            for agent, choice in enumerate(action): actions_by_agent[agent,choice] += 1
            obs, mask, reward, done, info = env.step(action)
            total += info['undiscounted_reward']
            durations[info['duration']] += 1
            for row in info['reward_components']:
                for key, value in row.items(): parts[key] += value
            if emit_macro is not None:
                emit_macro({'actions':action.tolist(), 'duration':info['duration'],
                            'reward_components':info['reward_components'],
                            'metrics':info['metrics']})
        return {**env.metrics(), 'scenario_id':scenario['scenario_id'], 'return':total,
                'macro_decisions':macros, 'policy_mode':'greedy', 'optimizer_updates':0,
                'action_counts_by_agent':actions_by_agent.tolist(),
                'macro_duration_counts':durations, 'reward_component_totals':parts}
    finally:
        env.close()
